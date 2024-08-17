

CUDA_VISIBLE_DEVICES=0 python ./pytorch-fid/src/pytorch_fid/fid_score.py /data/dailei/WHUGeoGen3/whugeogenv2_t2i_sdxl_bj_ny_512_epoch5/sample_512/WHUGeoGen_v2_test_data512_high_BeiJing_sample  /data/dailei/WHUGeoGen3/test/eval/test_data512_high_BeiJing_eval

CUDA_VISIBLE_DEVICES=0 python ./pytorch-fid/src/pytorch_fid/fid_score.py /data/dailei/WHUGeoGen3/whugeogenv2_t2i_sdxl_bj_ny_512_epoch5/sample_512/WHUGeoGen_v2_test_data512_mid_BeiJing_sample  /data/dailei/WHUGeoGen3/test/eval/test_data512_mid_BeiJing_eval

CUDA_VISIBLE_DEVICES=0 python ./pytorch-fid/src/pytorch_fid/fid_score.py /data/dailei/WHUGeoGen3/whugeogenv2_t2i_sdxl_bj_ny_512_epoch5/sample_512/WHUGeoGen_v2_test_data512_low_BeiJing_sample  /data/dailei/WHUGeoGen3/test/eval/test_data512_low_BeiJing_eval


CUDA_VISIBLE_DEVICES=0 python ./pytorch-fid/src/pytorch_fid/fid_score.py /data/dailei/WHUGeoGen3/whugeogenv2_t2i_sdxl_bj_ny_512_epoch5/sample_512/WHUGeoGen_v2_test_data512_high_NewYork_sample  /data/dailei/WHUGeoGen3/test/eval/test_data512_high_NewYork_eval

CUDA_VISIBLE_DEVICES=0 python ./pytorch-fid/src/pytorch_fid/fid_score.py /data/dailei/WHUGeoGen3/whugeogenv2_t2i_sdxl_bj_ny_512_epoch5/sample_512/WHUGeoGen_v2_test_data512_mid_NewYork_sample  /data/dailei/WHUGeoGen3/test/eval/test_data512_mid_NewYork_eval

CUDA_VISIBLE_DEVICES=0 python ./pytorch-fid/src/pytorch_fid/fid_score.py /data/dailei/WHUGeoGen3/whugeogenv2_t2i_sdxl_bj_ny_512_epoch5/sample_512/WHUGeoGen_v2_test_data512_low_NewYork_sample  /data/dailei/WHUGeoGen3/test/eval/test_data512_low_NewYork_eval