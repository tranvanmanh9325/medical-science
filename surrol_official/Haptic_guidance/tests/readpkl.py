import pickle 
# self.path 23 is: [-0.03378095 -0.15346667 -0.32213458 -0.04483975  0.58690107]
# self.path 24 is: [-0.0049948  -0.02120442  0.08207306  0.01046221 -0.63971394]
file = open('/home/kj/skjsurrol/SurRoL_skj/tests/saved_peg_transfer_action_psm.pkl','rb')
data = pickle.load(file)
file.close()
print('showing:')
idx = 0
for item in data:
    print('self.path',idx,'is:',item)
    idx += 1