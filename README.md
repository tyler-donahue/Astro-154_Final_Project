# CLASSIFICATION OF SUPERNOVA LIGHTCURVES FROM THE TRANSIENT NAME SERVER USING MACHINE LEARNING ALGORITHMS

With the introduction of Rubin Observatory, beginning of LSST, and exponentially increasing collection of observational data on transients, the development of autonomous transient classification pipelines will be of particular importance in the coming years. Type Ia SNe in particular comprise a fundamental component of the cosmic distance ladder, and as such, optimization of the classification and analysis of their lightcurves will prove essential to large-data cosmology in the coming decades. Presently, supernovae are primarily classified using spectroscopic data, rather than photometry, as photometric data typically carries higher uncertainty and lacks physical markers of supernova classification as utilized in spectroscopic classification. Consequently, the training of models to accurately classify by use of photometry alone, if viable, presents a highly efficient solution to the current large collection of photometry and minimal capacity to take spectroscopic observations by astronomers today. In this study, we utilize the packages numpy, scipy, pandas, matplotlib, scikit-learn, and george to construct classification pipelines for BLANK supernova photometric observations. We do this by using george to model fit light curves, and a random forest machine learning algorithm with cross-validation to test its accuracy. We find that, given our dataset, we were able to achieve a maximum accuracy of 67.45\%. Our algorithm was most successful at classifying Ib and Ic SNe, and was least adept at classifying Ia SNe.

## Repository Layout
* Code
* * Contains fully commented and implemented code for this project
* Data
* * Contains all data used and processed for this project
* Tests
* * Contains all test code and data for project - is uncommeted, unsorted, and incomplete.  If used in project was implemented fully in Code folder
