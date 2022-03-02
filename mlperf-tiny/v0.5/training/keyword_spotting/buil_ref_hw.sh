# suppress informational messages from TF
export TF_CPP_MIN_LOG_LEVEL=2 

export CUDA_VISIBLE_DEVICES=1

# python train.py --saved_model_path=trained_models/kws_ref_model \
#        --epochs=30 --run_test_set=True  
# python quantize.py --saved_model_path=trained_models/kws_ref_model.h5 \
#        --tfl_file_name=trained_models/kws_ref_model.tflite 
# python eval_quantized_model.py --tfl_file_name=trained_models/kws_ref_model.tflite 

python train.py --saved_model_path=trained_models/kws_dscnn-s_lfbe_filter --model_architecture=ds_cnn_flat \
       --epochs=30 --run_test_set=True --dct_coefficient_count=10 --feature_type=lfbe
