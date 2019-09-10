#!/bin/bash

# train baseline classifier
for index in {0..8} 
do 
	python train_baseline_classifier.py --index ${index}
done

# train MCL
for index in {0..95} 
do 
	python train_MCL.py --index ${index}
done

# train autoencoder
for index in {0..63} 
do 
	python train_autoencoder.py --index ${index}
done

# train prior
for index in {0..51} 
do 
	python train_prior.py --index ${index}
done

# train prior in semi-supervised setting
for index in {0..35} 
do 
	python train_prior_semi_supervised.py --index ${index}
done

# train MCLwP
for index in {0..575} 
do 
	python train_MCLwP.py --index ${index}
done

# train MCLwoP
for index in {0..59} 
do 
	python train_MCLwoP.py --index ${index}
done

# train MCLwPS
for index in {0..107} 
do 
	python train_MCLwPS.py --index ${index}
done

