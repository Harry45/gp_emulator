# GP Emulator For KiDS-450 Analysis 

We use the publicly available KiDS-450 data (see <a href="http://kids.strw.leidenuniv.nl/sciencedata.php">here</a>). 


### Computing $\sigma_{8}$

<p align="justify">Since our emulator is not a function of $\sigma_{8}$, we record the values of the latter as we generate our training set and build a GP emulator on top it. We also make this package publicly available and is found <a href="http://kids.strw.leidenuniv.nl/sciencedata.php">here</a>. In particular, the GP for $\sigma$ is a function of the following parameters:</p>

$$
\Omega_{\textrm{cdm}}h^{2},\,\Omega_{\textrm{b}}h^{2},\,\textrm{ln}\left(10^{10}A_{s}\right),\,n_{s},\,h,\,\Sigma m_{\nu}
$$
