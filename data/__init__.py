from data.data import OctDB
from data.utils import compute_knn, normalize, subdivide, get_cell_size


def get_dataset(args, split='training'):
    """ Get dataset """
    dataset_type = args.dataset_type
    if dataset_type is None:
        raise ValueError('Please provide a dataset type')

    elif dataset_type == 'OctDB':
        return OctDB(args, split)
    else:
        raise NotImplementedError('Dataset not supported {}'.format(dataset_type))
