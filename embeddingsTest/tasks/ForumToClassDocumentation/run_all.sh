python test_class_posts.py /data/blanca/class_posts_test_data.json USE /dev/null > USE.results

python test_class_posts.py /data/blanca/class_posts_test_data.json bert /dev/null > bert.results

python test_class_posts.py /data/blanca/class_posts_test_data.json xlm /dev/null > xlm.results

python test_class_posts.py /data/blanca/class_posts_test_data.json distilbert_para /dev/null > distilbert.results

python test_class_posts.py /data/blanca/class_posts_test_data.json msmacro /dev/null > msmarco.results

python test_class_posts.py /data/blanca/class_posts_test_data.json finetuned /data/BERTOverflow-tuned/class_posts/dccstor/m3/blanca/BERTOverflow/-2021-01-11_12-14-54/0_Transformer/ > tuned.results

python test_class_posts.py /data/blanca/class_posts_test_data.json bertoverflow /data/BERTOverflow > bertoverflow.results





