export LMUData=dataset


#********#
python -u run.py --data PhyX_mini_OE \
    --model InternVL2_5-8B \
    --judge-args '{"valid_type": "STR"}'


#********#
export SiliconFlow_API_KEY=$Your_key # for judger
python -u run.py --data PhyX_mini_OE \
    --model InternVL2_5-8B \
    --judge deepseek-v3-si --judge-args '{"valid_type": "LLM"}'


#********#
export Deepseek_API=$Your_key # for judger
python -u run.py --data PhyX_mini_OE \
    --model InternVL2_5-8B \
    --judge deepseek-v3 --judge-args '{"valid_type": "LLM"}'
    