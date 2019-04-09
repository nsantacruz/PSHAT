# PSHAT
(Pronounced "P'Shot") Part of Speech Handling for Aramaic Talmud

This is the official repo for Noah's Master's thesis.

This project aims to fill the gaping hole in ancient Aramaic POS tagging. Astonishingly, this field of research is scant. My work begins to show that modern machine learning techniques can learn patterns syntactic patterns in Talmud, despite two major issues

1. Talmud has no punctuation. Because of this, it can be very difficult to break up sentences and ideas, even if one is familiar with the Aramaic and the structure of the text

2. Talmud is actually a mix of two languages, Mishnaic Hebrew and Talmudic Aramaic. While in some places the distinction between these languages is clearly marked, the majority of Talmud is a mixture of the two. 

Despite these issues, LSTMs were able to achieve above 90% POS tagging on a validation set.  

I gratefully thank [CAL](http://cal1.cn.huc.edu/) and especially Steve Kaufman for working with me on this project. The use of his dataset was crucial and his help working with the dataset was just as important. 

## Requirements

1. This project uses the [Sefaria](https://sefaria.org) library. Certain scripts require you to have Sefaria set up on your computer. Follow the instructions on their [repo](https://github.com/Sefaria/Sefaria-Project) to set it up.

2. You need to install [dynet](https://github.com/clab/dynet) to run the LSTMs.

## Pipeline

1. `DatasetMatcher.py`: takes input from `data/1_cal_input` and outputs to `data/2_matched_sefaria`.
2. `LangDatasetGenerator.py`: generates language training dataset from Sefaria library and CAL files. Aramaic training comes from `data/1_cal_input/caldbfull.txt` and Mishnaic training comes from Sefaria's Mishnah. Outputs training as json file to `data/3_lang_tagged/model/lstm_training.json`. NOTE: This file isn't written perfectly. There's a bool at the bottom, `make_training`. If true, it generates training files. Otherwise, see step (4).
3. `LangTagger.py`: takes input from `data/3_lang_tagged/model/lstm_training.json` and trains an LSTM to differentiate between Hebrew and Aramaic (only on individual words). Outputs to `data/3_lang_tagged`.
4. Dilate language tagged output. Run `LangDatasetGenerator.py` with `make_training = False`. Outputs to `4_lang_tagged_dilated`
5. `POSTagger2MLP-beam.py`: takes input from `4_lang_tagged_dilated`, `2_sefaria_matched` and outputs to `5_pos_tagged`. Trains LSTM to learn POS tags of Aramaic words in Talmud
