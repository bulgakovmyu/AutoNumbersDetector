import subprocess
import sys

def install_dependences():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyyaml==5.1'])
    subprocess.check_call(['apt-get', 'install', 'tesseract-ocr'])
    subprocess.check_call(['apt-get', 'install',  'libtesseract-dev'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytesseract'])
    import torch
    assert torch.__version__.startswith("1.7")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'detectron2', '-f', 				    
    						'https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html'])
    print('All dependences are installed! You need to restart the environment!')
                            