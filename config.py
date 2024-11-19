import torch, os
import yaml
from IPython import embed


def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data)
    return data

import torch, os
import yaml
from IPython import embed


def get_yaml_data(yaml_file):
    with open(yaml_file, 'r', encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def get_config(args):
    configuration = dict(
        SEED=1337,  # random seed for reproduce results
        INPUT_SIZE=[112, 112],
        EMBEDDING_SIZE=512,  # embedding size
        DROP_LAST=True,
        WEIGHT_DECAY=5e-4,
        MOMENTUM=0.9,
    )

    # Cấu hình GPU
    if args.workers_id == 'cpu' or not torch.cuda.is_available():
        configuration['GPU_ID'] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration['GPU_ID'] = [int(i) for i in args.workers_id.split(',')]
    if len(configuration['GPU_ID']) == 0:
        configuration['DEVICE'] = torch.device('cpu')
        configuration['MULTI_GPU'] = False
    else:
        configuration['DEVICE'] = torch.device('cuda:%d' % configuration['GPU_ID'][0])
        if len(configuration['GPU_ID']) == 1:
            configuration['MULTI_GPU'] = False
        else:
            configuration['MULTI_GPU'] = True

    # Cấu hình huấn luyện
    configuration['NUM_EPOCH'] = args.epochs
    configuration['STAGES'] = [int(i) for i in args.stages.split(',')]
    configuration['LR'] = args.lr
    configuration['BATCH_SIZE'] = args.batch_size

    # Xử lý các chế độ dữ liệu khác nhau
    if args.data_mode == 'casia':
        configuration['DATA_ROOT'] = '/home/asus/Downloads/SFace-main/eval/lfw/'  # Thư mục cho huấn luyện
    elif args.data_mode == 'ms1m':
        configuration['DATA_ROOT'] = '/home/asus/Downloads/SFace-main/eval/lfw/'  # **Cập nhật đường dẫn này đến thư mục chứa train.rec và property**
    else:
        raise ValueError(f"Unsupported data_mode: {args.data_mode}. Supported modes are: 'casia', 'ms1m'.")

    # Cấu hình đường dẫn đánh giá
    configuration['EVAL_PATH'] = '/home/asus/Downloads/SFace-main/eval/lfw'  # Thư mục cho validation
    configuration['TARGET'] = ['lfw']
    # Xác minh loại mạng và head
    assert args.net in ['IR_50', 'IR_101', 'MobileFaceNet']
    configuration['BACKBONE_NAME'] = args.net

    assert args.head in ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'SFaceLoss']
    configuration['HEAD_NAME'] = args.head

    # Xác định các mục tiêu kiểm tra
    configuration['TARGET'] = [i for i in args.target.split(',')]

    # Xử lý việc tiếp tục huấn luyện từ checkpoint
    if args.resume_backbone:
        configuration['BACKBONE_RESUME_ROOT'] = args.resume_backbone  # Thư mục để tiếp tục huấn luyện từ checkpoint
        configuration['HEAD_RESUME_ROOT'] = args.resume_head  # Thư mục để tiếp tục huấn luyện từ checkpoint
    else:
        configuration['BACKBONE_RESUME_ROOT'] = ''
        configuration['HEAD_RESUME_ROOT'] = ''

    # Đường dẫn xuất cho kết quả
    configuration['WORK_PATH'] = args.outdir  # Thư mục để lưu checkpoints
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return configuration

