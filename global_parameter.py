thr_track=0.6
# thr_det = 0.4 if len(prior)==0  else 0.6
thr_det_global = 0.7
thr_matching_mask_label0 = 0.5
thr_matching_mask_label12 = 0.01
thr_pos_rm = 0.03
single_traversal = False
map_add,map_rm,map_change  = True,False,True
print('thr_track:',thr_track)
print('thr_det_global:',thr_det_global)
print('thr_matching_mask_label0:',thr_matching_mask_label0)
print('thr_matching_mask_label12:',thr_matching_mask_label12)
print('thr_pos_rm:',thr_pos_rm)
print('single_traversal:',single_traversal)
print('map_add:',map_add)
print('map_rm:',map_rm)
print('map_change:',map_change)