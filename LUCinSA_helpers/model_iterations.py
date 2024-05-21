#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
from .handler import logger
from LUCinSA_helpers.rf import balance_training_data
from LUCinSA_helpers.rf import get_class_col
from LUCinSA_helpers.rf import get_stable_holdout
from LUCinSA_helpers.rf import rf_model

def get_balanced_sample_for_sm_test(samp_pts, lut, model_dir, scratch_dir, cutoff):
    pts = pd.read_csv(samp_pts)
    pts.drop(['LC2','LC3','LC4','LC5','LC_UNQ'], axis=1, inplace=True)
    ptdf = pts.merge(pd.read_csv(lut), left_on='Class', right_on='USE_NAME', how='left')

    ## First balance classes so that holdout is representative of a random sample for that group.
    ##   THe method we are using leaves mixed classes oversampled so we can test effect later. 
    ##   Need to compensate for this in the ho

    ## pre-balance the soy, to make that we keep our highest quality samples:
    soyground = ptdf[(ptdf['LC25_name'] == 'Crops-Soybeans') & (ptdf['SampMethod'] != 'CAN - unverified in GE')]
    allsoy = ptdf['LC25_name'].value_counts()['Crops-Soybeans']
    othersoy = (1550 - soyground.shape[0]) / allsoy
    ptdf_balsoy = ptdf[(ptdf['rand'] < othersoy) | (ptdf['SampMethod'] != 'CAN - unverified in GE')]
    ## now balbnce all classes, but keep max mixed (to adjust later):
    ptdf_bal = balance_training_data(lut, ptdf_balsoy, scratch_dir, cutoff = cutoff, mix_factor = 10)
    ptdf_bal.drop(['Description', 'ratios','Segmentation','LCTrans','LCTrans_name'], axis=1, inplace=True)
    #sys.stderr.write(f'{ptdf_bal.columns.tolist()} \n')
    ptdf_bal.to_csv(os.path.join(model_dir,'ptdf_bal100mix10_2021.csv'),index=False)

def get_stable_holdout_for_sm_test(model_dir, samp_file, lut=None):
    fixed_ho_dir = os.path.join(model_dir,'fixed_HOs')
    ptdf = pd.read_csv(samp_file)
    ho, tr0 = get_stable_holdout(ptdf, fixed_ho_dir, 20, 'smallCrop', balanced_input=True, lut=None, overwrite=True) 
    ho1, tr1 = get_stable_holdout(tr0, fixed_ho_dir, 20, 'bigCrop', balanced_input=True, lut=None, overwrite=True) 
    ho2, tr2 = get_stable_holdout(tr1, fixed_ho_dir, 20, 'noCrop', balanced_input=True, lut=None, overwrite=True)
    
def iterate_mixed_sample_for_sm_test(n_mixcrop, feat_model, model_dir, scratch_dir):
    '''
    adds 10 smallholder samples per increment, as well as more Mixed-path and Mixed-VegEdge samples.
    '''
    tr = pd.read_csv(os.path.join(model_dir,'fixed_HOs','{}_GENERAL_TRAINING.csv'.format(feat_model)))
    allmc = tr['LC25_name'].value_counts()['Crops-mix']
    cutoff = (10 * mixcrop) / allmc
    #sys.stderr.write(f'training data had {len(tr)} records \n'.))
    training = tr[(tr['rand'] <= cutoff) | (tr['LC25_name'] != 'Crops-mix')] 
    training = training [(training['rand']<(nCrop/10)) | ((training['LC25_name'] != 'Mixed-path') 
                                                          & (training['LC25_name'] !='Mixed-VegEdge'))]
    num_mixed = pixdf['training'].value_counts()['Crops-mix']
    #sys.stderr.write(f'there are {num_mixed} mixed_crop points in the training data \n')
    #sys.stderr.write(f'training data now has {len(training)} records \n') )
    training.to_csv(os.path.join(model_dir, 'trainingF.csv'))
    feature_mod_dict = '/home/downspout-cel/paraguay_lc/Feature_Models.json'
    fixed_ho_dir = os.path.join(model_dir,'fixed_HOs')
    rf0 = rf_model(training,scratch_dir,'cropNoCrop','Impurity',23,'base4NoPoly_base1000',
                   lut,'base4NoPoly',0,feature_mod_dict,update_model_dict=False,fixed_ho=True,fixed_ho_dir=fixed_ho_dir)


def iterate_all_models_for_sm_test(sample_pts, model_dir, scratch_dir, lut, samp_model, class_models, feat_models, iterations=3, stop=1000, step=10, get_new_hos=False):
    
    ## get fixed holdout and maximum training sets
    samp_file = os.path.join(model_dir,f'ptdf_{samp_model}mix10_2021.csv')
    #sys.stderr.write(f'looking for {samp_file}\n' )
    if not os.path.exists(samp_file):
        sys.stderr.write('balancing full sample...\n' )
        #get_balanced_sample_for_sm_test(sample_pts, lut, model_dir, scratch_dir, cutoff=int(samp_model[:3]))
    fixed_ho_dir = os.path.join(model_dir,'fixed_HOs')
    if (get_new_hos == True) or (get_new_hos == 'True'):
        sys.stderr.write('getting new holdout samples...')
        get_stable_holdout_for_sm_test(model_dir,samp_file,lut=None)

    ## Feature models: 
    for fm in feat_models:
        logger.info(f'Working on feature model {fm}....\n')
        vardf = pd.read_csv('/home/downspout-cel/paraguay_lc/vector/ptsgdb_{}.csv'.format(fm))
        ## validate dataset:
        nancols =vardf.columns[vardf.isna().any()].tolist()
        if len(nancols) > 0:
            sys.stderr.write(f'ERROR - input dataset has feature columns with NaN: {nancols} \n')
        # make feature datasets from point files:
        for x in ['GENERAL_HOLDOUT_smallCrop','GENERAL_HOLDOUT_noCrop','GENERAL_HOLDOUT_bigCrop','GENERAL_TRAINING']:
            out_name = os.path.join(fixed_ho_dir,'{}.csv'.format(x.replace('GENERAL',fm)))
            if not os.path.exists(out_name):
                ptdf = pd.read_csv(os.path.join(fixed_ho_dir,f'{x}.csv'))
                pixdf = ptdf.merge(vardf, left_on='OID_', right_on='OID_', how='inner')
                pixdf.to_csv(out_name)
        
        ## Class_models:
        ##    (note, these are defined in rf.get_class_col)
        for lcmod in class_models:
            logger.info(f'Working on class model {lcmod}...\n')
            score_dict = {}
            class_mod = get_class_col(lcmod,lut)[0]
            logger.info(f'class column is: {class_mod}.\n')
            ## Sample models (adding in mixed)
            tr = pd.read_csv(os.path.join(fixed_ho_dir,'{}_TRAINING.csv'.format(fm)))
            allmc = tr['LC25_name'].value_counts()['Crops-mix']
            
            for n in range(stop/step):
                logger.info(f'iteration {n}...\n')
                ## get samples with rannum < cutoff val that would give ~<step> additional samples
                cutoff = (step * n) / allmc
                #sys.stderr.write(f'training data had {len(tr)} records \n'))
                training = tr[(tr['rand'] <= cutoff) | (tr['LC25_name'] != 'Crops-mix')] 
                training = training[(training['rand']<(n/100)) | ((training['LC25_name'] != 'Mixed-path') 
                                                                 & (training['LC25_name'] !='Mixed-VegEdge'))]
                #sys.stderr.write(f'training data now has {len(training)} records \n'))
                if n == 0:
                    num_mixed = 0
                else:
                    num_mixed = training['LC25_name'].value_counts()['Crops-mix']   
                #sys.stderr.write(f'there are {num_mixed} mixed_crop points in the training data \n')
                feature_mod_dict = '/home/downspout-cel/paraguay_lc/Feature_Models.json'
                score_dict[n]={}
                score_dict[n]['feat_model']=fm
                score_dict[n]['samp_model']=samp_model
                score_dict[n]['class_model']=class_mod
                score_dict[n]['n_small'] = num_mixed  
                score_dict[n]['acc_smallCrop']=[]
                score_dict[n]['acc_bigCrop']=[]
                score_dict[n]['acc_noCrop']=[]
                
                for iter in range(iterations):
                    sys.stderr.write(f'iteration {iter}...\n')
                    ran = 1 + 23*iter
                    rf0 = rf_model(training,scratch_dir,lcmod,'Impurity',ran,f'{fm}_{samp_model}',
                        lut,fm,0,feature_mod_dict,update_model_dict=False,fixed_ho=True,fixed_ho_dir=fixed_ho_dir)

                    score_dict[n]['acc_smallCrop'].append(round(rf0[1]['acc_smallCrop'],3))
                    score_dict[n]['acc_bigCrop'].append(round(rf0[1]['acc_bigCrop'],3))
                    score_dict[n]['acc_noCrop'].append(round(rf0[1]['acc_noCrop'],3))
    
                    with open(os.path.join(model_dir,f'model_iterations_{fm}_{class_mod}.json'), 'w', encoding="utf8") as out_file:
                        out_file.write( json.dumps(score_dict, default=int))
        
            mod_dict = json.loads(open(os.path.join(model_dir,'model_iterations_{}_{}.json'.format(fm,class_mod)),"r").read())
            df = pd.DataFrame.from_dict(mod_dict, orient='index')
            df['avgacc_smallCrop'] = df.apply(lambda x: round(np.mean(x['acc_smallCrop']),3), axis=1)
            df['avgacc_bigCrop'] = df.apply(lambda x: round(np.mean(x['acc_bigCrop']),3), axis=1)
            df['avgacc_noCrop'] = df.apply(lambda x: round(np.mean(x['acc_noCrop']),3), axis=1)
            df.to_csv(os.path.join(model_dir,f'smallCrop_iterations_{fm}_{class_mod}_{step}_{stop}.csv'))
        
