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

Interpolation between random points in latent space:

**in the past n years old 's n n stake in the first nine months of the company 's assets were n't disclosed**\
in the past n years old 's n n sales of $ n million or n cents a share from $ n million or $ n a share\
in the past year 's edition of the company 's assets were $ n billion in the first nine months of the year\
in the past year 's edition of the company 's assets were n't disclosed\
in addition the company said it will redeem $ n million in the first nine months of the year\
in addition the company said it will redeem $ n million in cash\
**in addition the company said it will be sold at $ n a share**

Sentences generate by the LSTM-VAE trained on YelpReview dataset:
*i had the <unk> enchilada combo with the taco . service was great and my food was good . i will return .\
this place was awesome . it was very dark and sweaty . the location was great , the food was good , and the service was great .\
great gluten free food . friendly staff and the prices were reasonable .\
wow , i do n't want to deal with <unk> ? ? ? ? ? ! ? i just ca n't really spend much of money on the slots and drink tickets when i get , i remember the golden nugget .\
the original con queso that you can get here are wonderful ! ! ! also love their chicken enchiladas with chips and salsa . the service is great and it is the place to go for a drink .\
i 've been coming here for years now . i have to say there is more than a place that draws its prices , but with good food . i rather go to mastro 's ocean or pin kaow or pin kaow .\
if you 're looking for a nice place to stay and not spend the extra money on a cheap ( this place is a great break .\
i had the mac and cheese , the brisket was dry and very dry . the tamale was <unk> , but the taste was bland . the sides were decent , but really loud . i have been to <unk> bbq places in san diego and since they were great , this place was a good vibe . i liked how it was nice , which was nice .\
the food is good . i usually order the pork bibimbap . it 's pretty good . the meat inside is perfect . the price point is right and if u get medium , you pay a lot . i had the teriyaki bowl thinking it was alright . the broth was flavorful but i did n't want to pay the price . the bok choy were also good . but i do n't think i 'll be coming back for the pho .*
