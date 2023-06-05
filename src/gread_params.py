import argparse
"""

"""
best_params_dict = {
'texas': {'reaction_term':'bspm', 'alpha_dim':'sc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 1.4607185193289032,'step_size': 1,
          'epoch':200, 'lr': 0.01882477235894058 ,'decay': 0.02466898106256942,
          'block':'constant', 'hidden_dim': 128 , 'data_norm':'gcn', 'self_loop_weight':0,
          'input_dropout': 0.4662723384814147, 'dropout': 0.4772766211800558 
           }, 
'wisconsin': {'reaction_term':'bspm', 'alpha_dim':'sc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'rk4', 'time': 1.7459655142602897,'step_size': 0.25,
          'epoch':200, 'lr': 0.01449628926673464 ,'decay': 0.00901220554965404,
          'block':'attention', 'hidden_dim': 256 , 'data_norm':'rw', 'self_loop_weight':0,
          'input_dropout': 0.5394876953124689, 'dropout': 0.4849875143400876,
          'XN_activation': True
           },
'cornell': {'reaction_term':'bspm', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'rk4', 'time': 0.11564783557675194,'step_size': 0.2,
          'epoch':200, 'lr': 0.008225031932075681 ,'decay': 0.028046982280138487,
          'block':'attention', 'hidden_dim': 128 , 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.48912249722614337, 'dropout': 0.3159670329306962,
          'use_mlp': True, 'XN_activation': False
           },
'film': {'reaction_term':'bspm', 'alpha_dim':'sc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'rk4', 'time': 0.30678550581567293,'step_size': 0.1,
          'epoch':200, 'lr': 0.007850471605952366 ,'decay': 0.0014341594128526826,
          'block':'attention', 'hidden_dim': 64 , 'data_norm':'rw', 'self_loop_weight':0,
          'input_dropout': 0.416670279482016, 'dropout': 0.6450760407185196,
          'use_mlp': True, 'XN_activation': False
           },
'squirrel': {'reaction_term':'bspm', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 5.696542814611847,'step_size':0.75,
          'epoch':200, 'lr': 0.01713941076746923 ,'decay': 2.8396246589758592e-06,
          'block':'constant', 'hidden_dim': 256, 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.5156955697758989, 'dropout': 0.09328362336851624,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': False
           },
'chameleon': {'reaction_term':'bspm', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 1.9439996002927,'step_size':1.5,
          'epoch':200, 'lr': 0.0067371581757143285 ,'decay': 7.736946152049231e-05,
          'block':'attention', 'hidden_dim': 256, 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.6759632513264229, 'dropout': 0.09328362336851624,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': True
           },
'Cora': {'reaction_term':'bspm', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'rk4', 'time': 3.790184078169178,'step_size':0.5,
          'epoch':200, 'lr': 0.011402915506754104 ,'decay': 0.008014968630105014,
          'block':'attention', 'hidden_dim': 64, 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.5043839651430236, 'dropout': 0.4145754297432822,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': True
           },
'Citeseer': {'reaction_term':'bspm', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'rk4', 'time': 2.0365995371213703,'step_size':0.2,
          'epoch':200, 'lr': 0.0029496654117168557 ,'decay': 0.013789766632941278,
          'block':'attention', 'hidden_dim': 128, 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.5224892802449188, 'dropout': 0.46161962752030056,
          'use_mlp': False, 'm2_mlp': False, 'XN_activation': True
           },
'Pubmed': {'reaction_term':'bspm', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'rk4', 'time': 1.736571888322607,'step_size':0.8,
          'epoch':200, 'lr': 0.010838870718586332 ,'decay': 0.0005182464582183332,
          'block':'attention', 'hidden_dim': 64, 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.3648432339951884, 'dropout': 0.25687002898139405,
          'use_mlp': True, 'm2_mlp': False, 'XN_activation': False
           },
}

def shared_grand_params(opt):
    opt['block'] = 'constant'
    opt['function'] = 'laplacian'
    opt['optimizer'] = 'adam'
    opt['epoch'] = 200
    opt['lr'] = 0.001
    opt['method'] = 'euler'
    opt['geom_gcn_splits'] = True
    return opt

def shared_gread_params(opt):
    opt['function'] = 'gread'
    opt['optimizer'] = 'adam'
    opt['geom_gcn_splits'] = True
    return opt

def hetero_params(opt):
    #added self loops and make undirected for chameleon & squirrel
    if opt['dataset'] in ['chameleon', 'squirrel']:
        opt['hetero_SL'] = True
        opt['hetero_undir'] = True
    return opt