nohup python -u __main__.py --data ../examples/csv/adult.csv --test_data ../examples/csv/adult.csv -m ../examples/csv/adult.json --load ../result/model.pth --save "../result/model.pth" -e 10 > ../result/output.log 2>&1 &
python __main__.py --data ../examples/csv/adult.csv --test_data ../examples/csv/adult.csv -m ../examples/csv/adult.json --load ../result/model.pth --save "../result/model.pth" -e 10
python __main__.py --data ../data/csv/breast/train_data.csv --test_data ../data/csv/breast/test_data.csv -m ../data/csv/breast/breast.json --save "../result/model.pth" -e 200


python acgan_distribution.py --num_epochs 300 --train_data breast/train_data.csv --test_data breast/test_data.csv