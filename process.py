# 因为要结合CPS，那么我们在第一阶段就使用到无标签数据，这个和原始ST++不一样，我们对无标签数据进行基本增强然后使用CPS训练
# 有标签数据
labeled_trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
# 无标签数据
unlabeled_trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.unlabeled_id_path)

labeled_trainset.ids = 2 * labeled_trainset.ids if len(labeled_trainset.ids) < 200 else labeled_trainset.ids
# 这个第0阶段的trainloader里面有有标签和无标签数据
# 有标签数据的dataloader
train_loader = DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0,
                          drop_last=True)
# 无标签数据的dataloader
unsupervised_train_loader = DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                       num_workers=0, drop_last=True)

model1, optimizer1 = init_basic_elems(args)

model2, optimizer2 = init_basic_elems(args)

print('\nParams: %.1fM' % count_params(model1))
print('\nParams: %.1fM' % count_params(model2))
# ===========================第1阶段使用有标签和无标签数据进行CPS训练============================================
# 第一阶段使用有标签数据进行训练，无标签数据进行伪监督
# train1(model1=None, model2=None, train_loader=None, unsupervised_train_loader=None, valloader=None, criterion=None,optimizer1=None, optimizer2=None, args=None)
best_model, checkpoints = train1(model1=model1, model2=model2, train_loader=train_loader,
                                 unsupervised_train_loader=unsupervised_train_loader, valloader=valloader,
                                 criterion=criterion, optimizer1=optimizer1, optimizer2=optimizer2, args=args)
print("finish SupOnly......")