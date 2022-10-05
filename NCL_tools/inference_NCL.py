import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize
import sys
sys.path.append('../')
from NCL_network1 import CatRSDNet1
from NCL_network2 import CatRSDNet2
from NCL_network3 import CatRSDNet3
from NCL_network4 import CatRSDNet4
from NCL_network5 import CatRSDNet5
#from models.catRSDNet_NL import CatRSDNet_NL
import glob
from utils.dataset_utils import DatasetNoLabel


def main(out, input, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- model
    assert os.path.isfile(checkpoint), checkpoint + ' is not a file.'



    # check which model is used CatNet or CatNet_NL
    checkpoint1 = torch.load('NCL1.pth', map_location='cpu')
    checkpoint2 = torch.load('NCL2.pth', map_location='cpu')
    checkpoint3 = torch.load('NCL3.pth', map_location='cpu')
    checkpoint4 = torch.load('NCL4.pth', map_location='cpu')
    checkpoint5 = torch.load('NCL5.pth', map_location='cpu')    

    model1 = CatRSDNet1().to(device)
    model1.set_cnn_as_feature_extractor()
    model1.load_state_dict(checkpoint1['model_dict'])
    model_type = 'CatNet'

    model2 = CatRSDNet2().to(device)
    model2.set_cnn_as_feature_extractor()
    model2.load_state_dict(checkpoint2['model_dict'])
    model_type = 'CatNet'

    model3 = CatRSDNet3().to(device)
    model3.set_cnn_as_feature_extractor()
    model3.load_state_dict(checkpoint3['model_dict'], False)
    model_type = 'CatNet'

    model4 = CatRSDNet4().to(device)
    model4.set_cnn_as_feature_extractor()
    model4.load_state_dict(checkpoint4['model_dict'], False)
    model_type = 'CatNet'

    model5 = CatRSDNet5().to(device)
    model5.set_cnn_as_feature_extractor()
    model5.load_state_dict(checkpoint5['model_dict'], False)
    model_type = 'CatNet'



    print(model_type, ' loaded')
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)
    model5 = model5.to(device)


    # find input format
    assert os.path.isdir(input), 'no valid input provided, needs to be a folder, run the process_video.py file first'
    video_folders = sorted(glob.glob(os.path.join(input, '*/')))
    if len(video_folders) == 0:
        video_folders = [input]

    for file in video_folders:
        #model.rnn.clear_state()
        vname = os.path.basename(os.path.dirname(file))
        # compute cnn features for whole video
        data = DatasetNoLabel(datafolders=[file], img_transform=Compose([ToPILImage(), Resize(224), ToTensor()]))
        dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        outputs = []
        print('start inference on ', vname)
        for ii, (X, elapsed_time, frame_no, rsd) in enumerate(dataloader):  # for each of the batches
            X = X.to(device)
            #model.rnn.set_batch(X.shape[0])
            elapsed_time = elapsed_time.unsqueeze(0).float().to(device)
            with torch.no_grad():
                if model_type == 'CatNet':
                    step_pred1, exp_pred1, rsd_pred1, s1 = model1(X, stateful=(ii > 0))
                    step_pred2, exp_pred2, rsd_pred2, s2 = model2(X, stateful=(ii > 0))
                    step_pred3, exp_pred3, rsd_pred3, s3 = model3(X, stateful=(ii > 0))
                    step_pred4, exp_pred4, rsd_pred4, s4 = model4(X, stateful=(ii > 0))
                    step_pred5, exp_pred5, rsd_pred5, s5 = model5(X, stateful=(ii > 0))

                    step_pred = (step_pred1 + step_pred2 + step_pred3 + step_pred4 + step_pred5) / 5
                    exp_pred = (exp_pred1 + exp_pred2 + exp_pred3 + exp_pred4 + exp_pred5) / 5
                    rsd_pred = (rsd_pred1 + rsd_pred2 + rsd_pred3 + rsd_pred4 + rsd_pred5) / 5
                    
                    
                    step_pred_hard = torch.argmax(step_pred.detach(), dim=-1).view(-1).cpu().numpy()
                    exp_pred_soft = exp_pred.clone().cpu().numpy()
                    exp_pred = (torch.argmax(exp_pred.detach(), dim=-1) + 1).view(-1).cpu().numpy()


                else: 
                    rsd_pred = model1(X, stateful=(ii > 0))
                    step_pred_hard = np.zeros(len(rsd_pred))  # dummy
                    exp_pred = np.zeros(len(rsd_pred))  # dummy
                    exp_pred_soft = np.zeros((len(rsd_pred), 2))  # dummy
                    step_pred_hard = torch.argmax(step_pred1, dim=-1).view(-1).cpu().numpy()
                    exp_pred_soft = exp_pred.clone().cpu().numpy()
                    exp_pred = (torch.argmax(exp_pred1, dim=-1) + 1).view(-1).cpu().numpy()

                progress_pred = elapsed_time/(elapsed_time+rsd_pred.T+0.00001)
                progress_pred = progress_pred.view(-1).cpu().numpy()
                rsd_pred = rsd_pred.view(-1).cpu().numpy()
                elapsed_time = elapsed_time.view(-1).cpu().numpy()
            outputs.append(np.asarray([elapsed_time, progress_pred, rsd_pred, step_pred_hard, exp_pred, exp_pred_soft[0], exp_pred_soft[1]]).T)
        outputs = np.concatenate(outputs).reshape(-1, 7)            
        np.savetxt(os.path.join(out, f'{vname}.csv'), outputs, delimiter=',', header='elapsed,progress,predicted_rsd,predicted_step,predicted_exp,predicted_assistant,predicted_senior', comments='')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--out',
        type=str,
        
        help='path to output folder.'
    )
    parser.add_argument(
        '--input',
        type=str,
        
        help='path to processed video file or multiple files.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        
        help='path to to model checkpoint .pth file.'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    main(out=args.out, input=args.input, checkpoint=args.checkpoint)

