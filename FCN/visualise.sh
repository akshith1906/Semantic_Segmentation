echo "8s"
python -m src.visualise --variant fcn8s
python -m src.visualise --variant fcn8s --freeze_backbone

echo && echo "16s"
python -m src.visualise --variant fcn16s
python -m src.visualise --variant fcn16s --freeze_backbone

echo && echo "32s"
python -m src.visualise --variant fcn32s
python -m src.visualise --variant fcn32s --freeze_backbone
