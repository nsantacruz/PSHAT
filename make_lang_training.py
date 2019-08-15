from LangDatasetGenerator import *

a = make_aramaic_training()
m = make_mishnaic_training()

full,lena,lenm,lenambig = merge_sets(a,m)

print 'A {} M {} Ambig {}'.format(lena,lenm,lenambig)
print 'Full Length: {}'.format(len(full))

fp = codecs.open("data/3_lang_tagged/model/lstm_training.json", "wb", encoding='utf-8')
json.dump(full, fp, indent=4, encoding='utf-8', ensure_ascii=False)
