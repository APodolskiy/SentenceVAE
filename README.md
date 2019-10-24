## VAE-based generative models

This repo is intended to reproduce some papers
on text generation with the usage of Variational
Autoencoder approach.

By now only  [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) by Bowman et al.
is implemented.

Results on PTB dataset.

| Split       | NLL   |
|:------------|:------:|
| Train       | 98.821 |
| Validation  | 103.2 |
| Test        | 103.9 |

Interpolation between random points in latent space (PTB dataset):

**in the past n years old 's n n stake in the first nine months of the company 's assets were n't disclosed**\
in the past n years old 's n n sales of $ n million or n cents a share from $ n million or $ n a share\
in the past year 's edition of the company 's assets were $ n billion in the first nine months of the year\
in the past year 's edition of the company 's assets were n't disclosed\
in addition the company said it will redeem $ n million in the first nine months of the year\
in addition the company said it will redeem $ n million in cash\
**in addition the company said it will be sold at $ n a share**

Sentences generate by the LSTM-VAE trained on YelpReview dataset with
temperature 0.8:

1. *i ate here on a livingsocial fest , and the food was pretty good . 
i had the chicken strips and the green chile soup , which i normally do n't like at all , 
but it was still very tasty . i would love to try a special new dish next time .*
2. *the wings were a good place to take the kids . the ingredients were really 
good and the service was awesome ! i 'd definitely come back here and try the red velvet .*
3. *i used to go here for years and it is decent . the staff is friendly and 
i get the simple drink , which is awesome . the only problem is i have noticed 
that the food is disgusting ! i wo n't go back .*
4. *i was introduced to the valley by one of the only one built in the area , 
and after trying a rock at the mirage , my flight decided to give it a shot and i know why . 
i have been to several cities , but like they were closing and seemed to be interested by the <unk> 
the staff was very helpful , and had a awesome environment , however , 
i would n't recommend this venue to anyone .*
5. *great location and great beer ! the patio is pretty cool too !*
6. *do n't waste your time . my boyfriend and i stopped by for dinner on a friday night in october . 
the weather was empty and the food was not great . the service was fine but the food was not good 
and the service was awful , maybe it was the servers fault us and they were n't busy . 
we probably wo n't be going back .*
