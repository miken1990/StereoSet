import glob
import json
from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib


PredictionStats = namedtuple('PredictionStats', ['num_erased_neurons', 'is_intrasentence', 'lm_score', 'ss_score', 'icat_score'])

def get_pred_stats_from_file(f_name):
    with open(f_name) as f:
        res = json.load(f)
        intrasentence_gender_info = res['gpt2-large']['intrasentence']['gender']
        num_erased_neurons = f_name.split('_')[-1].split('.')[0]

        intrasentence_stats = PredictionStats(
            num_erased_neurons=num_erased_neurons,
            is_intrasentence=True,
            lm_score=intrasentence_gender_info['LM Score'],
            ss_score=intrasentence_gender_info['SS Score'],
            icat_score=intrasentence_gender_info['ICAT Score']
        )

        intersentence_gender_info = res['gpt2-large']['intersentence']['gender']

        intersentence_stats = PredictionStats(
            num_erased_neurons=num_erased_neurons,
            is_intrasentence=False,
            lm_score=intersentence_gender_info['LM Score'],
            ss_score=intersentence_gender_info['SS Score'],
            icat_score=intersentence_gender_info['ICAT Score']
        )

        return intrasentence_stats, intersentence_stats


res_stats = []#/home/tzqmyp/nlp/StereoSet/code/scores/gpt2
for f_name in glob.glob('./scores/gpt2_large/*'):
    print(f_name)
    res_stats.append(get_pred_stats_from_file(f_name))

sorted_stats_list = sorted(res_stats, key=lambda x: x[0].num_erased_neurons)

x = range(1, 71)
lm_score = [x[0].lm_score for x in sorted_stats_list]
ss_score = [x[0].ss_score for x in sorted_stats_list]
icat_score = [x[0].icat_score for x in sorted_stats_list]


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(x, lm_score)
ax1.set_title('LM Score')

ax2.plot(x, ss_score)
ax2.set_title('SS Score')

ax3.plot(x, icat_score)
ax3.set_title('ICAT Score')

plt.show()





# lm_score = [x[1].lm_score for x in sorted_stats_list]
# ss_score = [x[1].ss_score for x in sorted_stats_list]
# icat_score = [x[1].icat_score for x in sorted_stats_list]
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# ax1.plot(x, lm_score)
# ax1.set_title('LM Score')
#
# ax2.plot(x, ss_score)
# ax2.set_title('SS Score')
#
# ax3.plot(x, icat_score)
# ax3.set_title('ICAT Score')
#
# plt.show()
# plt.plot(x, lm_score)
# plt.title('lm_score')
# #
# # plt.plot(x, ss_score)
# # plt.title('ss_score')
#
# # plt.plot(x, icat_score)
# # plt.title('icat_score')
# plt.show()




