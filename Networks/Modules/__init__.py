from Networks.Modules.HeadModules.RLHead import RLHeadModuleTSP, RLHeadModule_agg_before, RLHeadModule_agg_after
from Networks.Modules.HeadModules.NormalHead import NormalHeadModule,TransformerHead
### TODO implement mixture of AnnealedNoise and Bernoulli Noise
OutputHead_registry = {"RLHead": RLHeadModule_agg_before, "RLHead_aggr": RLHeadModule_agg_after, "RLHeadTSP": RLHeadModuleTSP, "NormalHead": NormalHeadModule, "TransformerHead": TransformerHead}
