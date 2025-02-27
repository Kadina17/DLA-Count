# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'MBM':
        from crowd_datasets.MBM.loading_data import loading_data
        return loading_data
    if args.dataset_file == 'DCC':
        from crowd_datasets.DCC.loading_data import loading_data
        return loading_data
    if args.dataset_file == 'ADI':
        from crowd_datasets.ADI.loading_data import loading_data
        return loading_data
    if args.dataset_file == 'VGG':
        from crowd_datasets.VGG.loading_data import loading_data
        return loading_data
    return None