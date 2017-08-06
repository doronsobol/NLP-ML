#Progress#

##Insights##
1. When using more complecated network on Bag-of-Words we get an overfit.
2. BOW is inclined to "agree" more than to "disagree".
3. A possible reason for (1) and (2) is that [I dont understand understood.]

##Ideas##
* Use Bi-LSTM between the sentance and its reference.
* Use LSTM on the closest sentence (using cosine similarity on BOW).
* Check results using Google.

