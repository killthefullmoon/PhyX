export PHYX_TEXT_ONLY=true
export LMUData=dataset
export Deepseek_API=$Your_key1 # for evaluated model


#********#
python -u run.py --data PhyX_mini_TL_OE \
    --model deepseek-r1 \
    --judge-args '{"valid_type": "STR"}'


#********#
export SiliconFlow_API_KEY=$Your_key # for judger
python -u run.py --data PhyX_mini_TL_OE \
    --model deepseek-r1 \
    --judge deepseek-v3-si --judge-args '{"valid_type": "LLM"}'


#********#
python -u run.py --data PhyX_mini_TL_OE \
    --model deepseek-r1 \
    --judge deepseek-v3 --judge-args '{"valid_type": "LLM"}'