# adversarial-neural-crypt
Implementation of model described in 'Learning to Protect Communications with Adversarial Neural Cryptography' (Martín Abadi &amp; David G. Andersen, 2016, https://arxiv.org/abs/1610.06918)

This fork aims at studying the non-linearity of the produced lookup table plaintext-ciphertext for Alice. We first build the three networks (Alice, Bob and Eve) and then train Alice's network with a static value (binary of 8bits) to retrieve the correspond output (ciphertext of 8bits).


