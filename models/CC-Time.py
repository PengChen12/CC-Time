import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models.ReVIN import RevIN
from utils.CKA import CKA,CudaCKA
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel, GPT2Tokenizer
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from utils.tools import Decomposition,positional_encoding
import torch.nn.functional as F
from layers.layer import Transpose, Bank_MultiheadAttention, _MultiheadAttention, TSTEncoderLayer



class CC_Time(nn.Module):
    def __init__(self, configs):
        super(CC_Time, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        if self.stride > 1 or self.patch_size > 1:
            self.patch_num += 1

        self.revin_layer = RevIN(1, affine=True, subtract_last=False)
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_num, d_model=self.patch_d_model)

        ##time series Branch
        self.n_heads = configs.n_heads
        self.ts_layer_num = configs.ts_layer_num
        self.ts_d_model = configs.ts_d_model
        self.ts_d_ff = configs.ts_d_ff
        self.patch_in_layer = nn.Linear(configs.patch_size, self.ts_d_model)
        self.ts_branch_layers = nn.ModuleList(
            [TSTEncoderLayer(self.patch_num, self.patch_d_model, n_heads=self.n_heads, d_k=None, d_v=None,
                             d_ff=self.patch_d_ff, norm='BatchNorm',
                             attn_dropout=0.1, dropout=0.2, res_attention=False, pre_norm=False, store_attn=False)
             for i in range(self.patch_layer_num)])


        ##LLM Branch
        self.llm_d_model = configs.llm_d_model
        self.llm_d_ff = configs.llm_d_ff
        self.gpt_layers = configs.gpt_layers
        self.llm_in_layer = nn.Linear(configs.patch_size, configs.llm_d_model)
        self.llm_in_layer2 = nn.Linear(configs.seq_len, configs.llm_d_model)
        self.llm_temporal_layers = GPT2Model.from_pretrained('./../gpt2', output_attentions=True,
                                              output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.llm_temporal_layers.h = self.llm_temporal_layers.h[:configs.gpt_layers]

        self.llm_variable_layers = GPT2Model.from_pretrained('./../gpt2', output_attentions=True,
                                              output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.llm_variable_layers.h = self.llm_variable_layers.h[:configs.gpt_layers]
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            './../gpt2',
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.proj_layer1 = nn.Linear(configs.d_model, self.d_ff)
        self.proj_layer2 = nn.Linear(self.patch_d_model, configs.d_model)
        self.dropout = nn.Dropout(0.2)

        self.dataset_name = configs.dataset_name
        variable_description_list = {"ETTm1":33, "ETTm2":33, "ETTh1":33, "ETTh2":33, "weather":61, "ZafNoo":75, "CzeLan":68}
        self.text_len_Linear = nn.Linear(variable_description_list[self.dataset_name], 1)

        self.variable_num = configs.variable_num
        self.left_linear_layer = nn.Linear(1, self.variable_num)
        self.right_linear_layer = nn.Linear(1, self.variable_num)
        self.bank_dim = configs.bank_dim
        self.alpha = configs.alpha
        self.extractor = nn.Parameter(torch.randn(configs.enc_in, self.bank_dim), requires_grad=True)
        self.extractor_linear = nn.Linear(configs.d_model, self.bank_dim)
        self.extractor_attention = Bank_MultiheadAttention(self.bank_dim, 1, 64, 64, attn_dropout=0.1, proj_dropout=0.1,
                                                      res_attention=False)


        self.llm_to_ts = nn.Linear(configs.llm_d_model, self.ts_d_model)
        self.fusion_attention = nn.ModuleList([_MultiheadAttention(self.ts_d_model, self.n_heads, self.ts_d_model // self.n_heads,
                                                           self.ts_d_model // self.n_heads, attn_dropout=0.1, proj_dropout=0.1,
                                                           res_attention=False) for i in range(2)])
        self.current_memory_attention = _MultiheadAttention(self.ts_d_model, self.n_heads, self.ts_d_model // self.n_heads, self.ts_d_model // self.n_heads, attn_dropout=0.1, proj_dropout=0.1, res_attention=False)
        if configs.cross_norm == "BatchNorm":
            self.fusion_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.ts_d_model), Transpose(1,2)) for i in range(self.ts_layer_num)])
        else:
            self.fusion_norm = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.ts_d_model)) for i in range(self.ts_layer_num)])


        self.llm_out_layer = nn.Linear(self.d_ff * self.patch_num, configs.pred_len)
        self.ts_out_layer = nn.Linear(self.patch_d_model*self.patch_num, configs.pred_len)
        self.llm_variable_out_layer = nn.Linear(self.d_ff, configs.pred_len)


        for i, (name, param) in enumerate(self.llm_temporal_layers.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for i, (name, param) in enumerate(self.llm_variable_layers.named_parameters()):
            if 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False




    def forward(self, x, y, *args, **kwargs):
        B, L, M = x.shape
        output_dic = {}
        x = self.revin_layer(x, 'norm')
        x = rearrange(x, 'b l m -> b m l')
        llm_variable_embedding = self.llm_in_layer2(x)


        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        llm_temporal_embedding = self.llm_in_layer(x)
        x = rearrange(x, 'b m n p -> (b m) n p')


        if "ETT" in self.dataset_name:
            prompt = ["HUFL (High UseFul Load): Represents the useful power or load of the system or equipment under high load conditions during the recorded time period",
                     "HULL (High UseLess Load): Represents the useless power or load of the system or equipment under high load conditions during the recorded time period",
                     "MUFL (Middle UseFul Load): Represents the useful power or load of the system or equipment under medium load conditions during the recorded time period",
                     "MULL (Middle UseLess Load): Represents the useless power or load of the system or equipment under medium load conditions during the recorded time period",
                     "LUFL (Low UseFul Load): Represents the useful power or load of the system or equipment under low load conditions during the recorded time period",
                     "LULL (Low UseLess Load): Represents the useless power or load of the system or equipment under low load conditions during the recorded time period",
                     "OT (Oil Temperature): The target variable representing the oil temperature during the recorded time period. The goal of the prediction model is to predict this variable based on other variables"]

        elif 'ETTm2' in self.dataset_name:
            prompt = ["High Use Full Load (HUFL): This term indicates that the transformer is operating at or near its maximum designed capacity under high demand conditions. The transformer is working at full capacity, which can lead to increased heat generation and, consequently, a rise in oil temperature",
                      "High Use Less Load (HULL): This indicates a situation where the transformer is in a high-demand area but operating under a less loaded condition. Even though it's not at full load, the transformer may still need to operate at a relatively high level, which could impact the oil temperature",
                      "Middle Use Full Load (MUFL): This refers to the transformer operating at a moderate load level but still close to its full capacity. Although the load is not as heavy as in the HUFL state, the transformer may still generate a significant amount of heat, affecting the oil temperature",
                      "Middle Use Less Load (MULL): This term describes the operational state of a transformer in a medium-demand area during off-peak hours or when the load is lighter. The transformer is not under full load, but it may still operate at a higher level than in low-demand scenarios, potentially affecting the oil temperature",
                      "Low Use Full Load (LUFL): This term suggests that the transformer is operating at its full capacity during low-demand periods or in areas with low power consumption. Despite being at full load, the overall heat generation and oil temperature might be lower compared to HUFL or MUFL due to the inherently lower demand",
                     "Low Use Less Load (LULL): This refers to a transformer operating under light load conditions in low-demand areas or during off-peak times. The transformer's oil temperature is likely to be lower in this state due to reduced heat generation from the lower load",
                      "OT (Oil Temperature): The target variable representing the oil temperature during the recorded time period. The goal of the prediction model is to predict this variable based on other variables"]

        elif self.dataset_name == "weather":
            prompt = ["p (mbar): Atmospheric pressure measured in millibars. Atmospheric pressure is an important parameter in meteorology, influencing weather and climate changes",
                      "T (degC): Ambient temperature measured in degrees Celsius. Temperature is a fundamental parameter describing climate conditions, affecting meteorological phenomena and biological activities",
                      "pot (K): Potential temperature, the temperature an air parcel would have if it were expanded or compressed adiabatically to a standard pressure (usually 1000 hPa), measured in Kelvin. Potential temperature is used to study the thermal properties and stability of air parcels",
                      "Tdew (degC): Dew point temperature, the temperature at which air becomes saturated with moisture and dew can form, measured in degrees Celsius. Dew point temperature is used to measure the humidity of the air",
                      "rh (%): Relative humidity, representing the percentage of moisture in the air relative to the maximum amount the air can hold at that temperature. Relative humidity describes the moisture content of the air",
                      "VPmax (mbar): Maximum vapor pressure, the pressure at which water vapor is in equilibrium with its liquid or solid form at the current temperature, measured in millibars. Maximum vapor pressure is an important parameter for calculating relative humidity and dew point temperature",
                      "VPact (mbar): Actual vapor pressure, the partial pressure of water vapor in the air, measured in millibars. Actual vapor pressure describes the actual amount of water vapor in the air",
                      "VPdef (mbar): Vapor pressure deficit, the difference between the maximum vapor pressure and the actual vapor pressure, measured in millibars. Vapor pressure deficit measures the drying power of the air",
                      "sh (g/kg): Specific humidity, the mass of water vapor per unit mass of air, measured in grams per kilogram. Specific humidity describes the moisture characteristics of the air",
                      "H2OC (mmol/mol): Water vapor concentration, the molar concentration of water vapor in the air, measured in millimoles per mole. Water vapor concentration is used to study the amount of water vapor in the air",
                      "rho (g/m³): Air density, the mass of air per unit volume, measured in grams per cubic meter. Air density describes the density of the air and affects the dynamics of the atmosphere",
                      "wv (m/s): Wind velocity, the speed of the wind, measured in meters per second. Wind velocity describes the strength of the wind and its impact on meteorological phenomena",
                      "max. wv (m/s): Maximum wind velocity, the highest wind speed recorded, measured in meters per second. Maximum wind velocity describes extreme wind speed events and their impact",
                      "wd (deg): Wind direction, the direction from which the wind is blowing, measured in degrees. Wind direction describes the direction and path of the wind",
                      "rain (mm): Amount of rainfall, the depth of rain that falls over a given area, measured in millimeters. Rainfall amount describes the intensity and total amount of precipitation",
                      "raining (s): Duration of rainfall, the total time it rained, measured in seconds. Rainfall duration is used to study the temporal characteristics of precipitation events",
                      "SWDR (W/m²): Shortwave downward radiation, the amount of solar radiation reaching the Earth's surface, measured in watts per square meter. Shortwave radiation describes the input of solar energy",
                      "PAR (µmol/m²/s): Photosynthetically active radiation, the portion of light (400-700 nm) usable by plants for photosynthesis, measured in micromoles per square meter per second. Photosynthetically active radiation is used to study conditions for plant growth and photosynthesis",
                      "max. PAR (µmol/m²/s): Maximum photosynthetically active radiation, the highest value of PAR recorded, measured in micromoles per square meter per second. Maximum photosynthetically active radiation is used to study extreme light conditions and their impact",
                      "Tlog (degC): Logged temperature, a recorded value of temperature, measured in degrees Celsius. Logged temperature is used for historical temperature data preservation and analysis",
                      "OT: Target variable, which could be temperature or another specific meteorological parameter that the prediction model aims to forecast. The target variable is the object of prediction for the time series prediction model"]

        elif self.dataset_name == "CzeLan":
            prompt =["CZE_LAN_Cbe_Jt_16: This variable likely represents sap flow measurements. It could be in units of sap flow rate, such as grams per hour (g/h) or liters per hour (L/h), indicating the volume of sap flowing through a given area of the plant over a specific period",
                     "ta: Air temperature, typically measured in degrees Celsius (°C). This variable reflects the ambient temperature surrounding the plants",
                     "rh: Relative humidity, usually expressed as a percentage (%). It indicates the amount of moisture in the air relative to the maximum amount of moisture the air can hold at a given temperature",
                     "sw_in: Shortwave incoming radiation, usually measured in watts per square meter (W/m²). It quantifies the amount of solar radiation reaching the surface",
                     "ppfd_in: Photosynthetic photon flux density incoming, typically measured in micromoles per square meter per second (µmol/m²/s). This measures the amount of light available for photosynthesis",
                     "ws: Wind speed, usually measured in meters per second (m/s) or kilometers per hour (km/h). This variable indicates the speed of the wind affecting the plants",
                     "precip: Precipitation, measured in millimeters (mm). This variable quantifies the amount of rainfall received over a certain period",
                     "swc_shallow: Shallow soil water content, typically measured in volumetric percentage (%). It represents the amount of water present in the shallow soil layer",
                     "swc_deep: Deep soil water content, also measured in volumetric percentage (%). It indicates the amount of water in the deeper soil layers",
                     "ext_rad: Extraterrestrial radiation, usually measured in watts per square meter (W/m²). It quantifies the solar radiation received at the top of the Earth's atmosphere",
                     "OT (vpd): Vapor pressure deficit, typically measured in kilopascals (kPa). It represents the difference between the amount of moisture in the air and how much moisture the air can hold when it is saturated. This variable is crucial for understanding plant transpiration rates"]


        elif self.dataset_name == "ZafNoo":
            prompt = ["ZAF_NOO_E3_IRR_Mdo_Jt_2: This variable likely represents sap flow measurements. It could be in units of sap flow rate, such as grams per hour (g/h) or liters per hour (L/h), indicating the volume of sap flowing through a given area of the plant over a specific period",
                      "ta: Air temperature, typically measured in degrees Celsius (°C). This variable reflects the ambient temperature surrounding the plants",
                      "rh: Relative humidity, usually expressed as a percentage (%). It indicates the amount of moisture in the air relative to the maximum amount of moisture the air can hold at a given temperature",
                      "sw_in: Shortwave incoming radiation, usually measured in watts per square meter (W/m²). It quantifies the amount of solar radiation reaching the surface",
                      "ppfd_in: Photosynthetic photon flux density incoming, typically measured in micromoles per square meter per second (µmol/m²/s). This measures the amount of light available for photosynthesis",
                      "ws: Wind speed, usually measured in meters per second (m/s) or kilometers per hour (km/h). This variable indicates the speed of the wind affecting the plants",
                      "precip: Precipitation, measured in millimeters (mm). This variable quantifies the amount of rainfall received over a certain period",
                      "swc_shallow: Shallow soil water content, typically measured in volumetric percentage (%). It represents the amount of water present in the shallow soil layer",
                      "swc_deep: Deep soil water content, also measured in volumetric percentage (%). It indicates the amount of water in the deeper soil layers",
                      "ext_rad: Extraterrestrial radiation, usually measured in watts per square meter (W/m²). It quantifies the solar radiation received at the top of the Earth's atmosphere",
                      "OT (vpd): Vapor pressure deficit, typically measured in kilopascals (kPa). It represents the difference between the amount of moisture in the air and how much moisture the air can hold when it is saturated. This variable is crucial for understanding plant transpiration rates"]

        else:
            prompt = None




        if prompt == None:
            llm_x = llm_variable_embedding
        else:
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
            prompt_embeddings = self.llm_temporal_layers.get_input_embeddings()(prompt.to(x.device))  # (variable_num, prompt_token, dim)
            prompt_embeddings = self.text_len_Linear(prompt_embeddings.permute(0,2,1)).permute(0,2,1).squeeze(dim=1)
            llm_x = prompt_embeddings + llm_variable_embedding


        extractor_llm_x = self.extractor_linear(llm_x)

        bank_output, bank_weights = self.extractor_attention(extractor_llm_x, self.extractor, self.extractor) # bank_weights:[bs, 1, num_nodes, num_nodes]


        llm_temporal_outputs = rearrange(llm_temporal_embedding, 'b m n p -> (b m) n p')
        llm_temporal_hiddens = self.llm_temporal_layers(inputs_embeds=llm_temporal_outputs).hidden_states
        llm_temporal_outputs = llm_temporal_hiddens[-1]
        llm_temporal_outputs = self.proj_layer1(llm_temporal_outputs)
        llm_temporal_outputs = self.dropout(llm_temporal_outputs)


        llm_variable_outputs = llm_x
        llm_variable_hiddens = self.llm_variable_layers(inputs_embeds=llm_variable_outputs, alpha=self.alpha, correlation_attn_weights=bank_weights).hidden_states
        llm_variable_outputs = llm_variable_hiddens[-1]


        ts_outputs = self.patch_in_layer(x) + self.W_pos

        llm_variable_hiddens_1 = rearrange(llm_variable_hiddens[1], 'b m p -> (b m) 1 p')
        llm_variable_hiddens_1 = llm_variable_hiddens_1.repeat(1, self.variable_num, 1)
        llm_memory_hidden = torch.cat([llm_variable_hiddens_1, llm_temporal_hiddens[1], llm_variable_hiddens_1], dim=1)
        llm_memory_hidden = self.llm_to_ts(llm_memory_hidden)



        for i in range(self.ts_layer_num):
            if i == 0:
                llm_variable_hiddens_i = llm_variable_hiddens[2*i+1]
                llm_variable_hiddens_i = rearrange(llm_variable_hiddens_i, 'b m p -> (b m) 1 p')
                llm_variable_hiddens_i = llm_variable_hiddens_i.repeat(1, self.variable_num, 1)
                llm_current_hidden = torch.cat([llm_variable_hiddens_i, llm_temporal_hiddens[2 * i + 1], llm_variable_hiddens_i], dim=1)
                llm_current_hidden = self.llm_to_ts(llm_current_hidden)
                llm_ts_fusion_outputs = self.fusion_attention[0](ts_outputs, llm_current_hidden, llm_current_hidden)[0]
                llm_ts_fusion_outputs = self.fusion_norm[i](llm_ts_fusion_outputs)

                ts_outputs = self.ts_layers[i](ts_outputs, ts_outputs)
                ts_outputs = llm_ts_fusion_outputs+ts_outputs

            else:
                llm_variable_hiddens_i = llm_variable_hiddens[2*i+1]
                llm_variable_hiddens_i = rearrange(llm_variable_hiddens_i, 'b m p -> (b m) 1 p')
                llm_variable_hiddens_i = llm_variable_hiddens_i.repeat(1, self.variable_num, 1)
                llm_current_hidden = torch.cat([llm_variable_hiddens_i, llm_temporal_hiddens[2 * i + 1], llm_variable_hiddens_i], dim=1)
                llm_current_hidden = self.llm_to_ts(llm_current_hidden)
                self.bias1.to(llm_current_hidden.device)
                self.bias2.to(llm_current_hidden.device)
                llm_current_hidden = (self.bias1*self.current_memory_attention(llm_current_hidden, llm_memory_hidden, llm_memory_hidden)[0] +
                              self.bias2*self.current_memory_attention(llm_memory_hidden, llm_current_hidden, llm_current_hidden)[0])


                llm_ts_fusion_outputs = self.fusion_attention[0](ts_outputs, llm_current_hidden, llm_current_hidden)[0]
                llm_ts_fusion_outputs = self.fusion_norm[i](llm_ts_fusion_outputs)

                ts_outputs = self.ts_layers[i](ts_outputs, ts_outputs)
                ts_outputs = llm_ts_fusion_outputs+ts_outputs

                llm_memory_hidden = llm_current_hidden


        llm_outputs = llm_temporal_outputs
        llm_outputs = self.llm_out_layer(llm_outputs.reshape(B * M, -1))
        llm_outputs = rearrange(llm_outputs, '(b m) l -> b l m', b=B) + self.gpt_variable_out_layer(llm_variable_outputs).permute(0,2,1)
        llm_outputs = self.revin_layer(llm_outputs, 'denorm')



        ts_outputs = self.ts_out_layer(ts_outputs.reshape(B * M, -1))
        ts_outputs = rearrange(ts_outputs, '(b m) l -> b l m', b=B)
        ts_outputs = self.revin_layer(ts_outputs, 'denorm')

        output_dic["llm_outputs"] = llm_outputs
        output_dic["ts_outputs"] = ts_outputs
        return output_dic

