# PSHAT
(Pronounced "P'Shot") Part of Speech Handling for Aramaic Talmud

This is the official repo for Noah's Master's thesis, currently in Beta. You're welcome to help Noah finish his thesis by forking and submitting a pull request. Or you can wait for the full release.

This project aims to fill the gaping whole in ancient Aramaic POS tagging. Astonishingly, this field of research is scant. My work begins to show that modern machine learning techniques can learn patterns syntactic patterns in Talmud, despite two major issues

1. Talmud has no punctuation. Because of this, it can be very difficult to break up sentences and ideas, even if one is familiar with the Aramaic and the structure of the text

2. Talmud is actually a mix of two languages, Mishnaic Hebrew and Talmudic Aramaic. While in some places the distinction between these languages is clearly marked, the majority of Talmud is a mixture of the two. 

Despite these issues, LSTMs were able to achieve above 90% POS tagging on a validation set.  

I gratefully thank [CAL] (http://cal1.cn.huc.edu/) and especially Steve Kaufman for working with me on this project. The use of his dataset was crucial and his help working with the dataset was just as important. 
