import sys
import traceback
sys.path.insert(0, '.')
from smart_city.pipeline import SmartCityPipeline
import os

PROJECT_ROOT = os.getcwd()
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'smart_city', 'config')
SEG_WEIGHTS = os.path.join(PROJECT_ROOT, 'pretrained_weights', 'sfpnr50.pth')
VQA_WEIGHTS = os.path.join(PROJECT_ROOT, 'pretrained_weights', 'soba.pth')

print("Starting debug script...")
try:
    p = SmartCityPipeline(
        config_dir=CONFIG_DIR,
        seg_weights_path=SEG_WEIGHTS,
        vqa_weights_path=VQA_WEIGHTS,
    )
    print("Pipeline initialized. Calling analyze_image...")
    result = p.analyze_image('4191.png')
    print("SUCCESS!")
except Exception as e:
    print("FAILED!")
    traceback.print_exc()
