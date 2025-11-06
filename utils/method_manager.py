import logging

from methods.bic import BiasCorrection
from methods.er_baseline import ER
from methods.rainbow_memory import RM
from methods.ewc import EWCpp
from methods.mir import MIR
from methods.clib import CLIB
from methods.gdumb import GDumb
from methods.gdumb import GDumb
from methods.er_baseline_nfm import ER_NFM
from methods.er_baseline_no_lr_schedule import ER_No_LR_Schedule
from methods.er_baseline_multi_swag import ER_Multi_SWAG
from methods.dro import DRO


logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes, writer):
    kwargs = vars(args)

    if args.mode == "er":
        method = ER(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )        
    
    elif args.mode == "ewc++":
        method = EWCpp(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )        
        
    elif args.mode == "bic":
        method = BiasCorrection(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )    
        
    elif args.mode == "mir":
        method = MIR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )    
        
    elif args.mode == "rm":
        method = RM(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )        
        
    elif args.mode == "gdumb":
        method = GDumb(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )

    elif args.mode == "clib":
        method = CLIB(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )          
        
    elif args.mode == "fms":
        method = FMS(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )      
        
    elif args.mode == "er_nfm":
        method = ER_NFM(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )                    
        
    elif args.mode == "er_no_lr_schedule":
        method = ER_No_LR_Schedule(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )                    
            
    elif args.mode == "er_multi_swag":
        method = ER_Multi_SWAG(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )                    
            
        
    elif args.mode == "dro":
        method = DRO(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            writer=writer,
            **kwargs,
        )                    
    
        
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method
