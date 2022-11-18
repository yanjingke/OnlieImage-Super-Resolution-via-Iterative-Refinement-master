'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase,sampler=None):
    '''create dataloader '''
    if phase == 'train':
        if sampler==None:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=dataset_opt['use_shuffle'],
                num_workers=dataset_opt['num_workers'],
                # shuffle=False,

                # sampler=sampler,
                pin_memory=True)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            # shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            shuffle=False,

            sampler=sampler,
            pin_memory=True)
    elif phase == 'val':
        if sampler==None:
            return torch.utils.data.DataLoader(
            dataset, batch_size=1,  shuffle=False,
                          pin_memory=True)
        return torch.utils.data.DataLoader(
            dataset, batch_size=1,  shuffle=False,
                          pin_memory=True,
                          sampler=sampler)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))

def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
