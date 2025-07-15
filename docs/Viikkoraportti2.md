## Viikkoraportti 2

### Mitä olen tehnyt tällä viikolla?
Tällä viikolla olen:
* Suunnitellut neuroverkon rakenteen
* Aloittanut projektin ydinalueen kehityksen
* Toteuttanut neuroverkon eteenpäin kytkennän

### Miten ohjelma on edistynyt?
Tällä hetkellä neuroverkko toimii yhteen suuntaan, eli syötteet kulkevat jo neuroverkon läpi, mutta vastavirta-algoritmi uupuu vielä.

### Mitä opin tällä viikolla?
Neuroverkon gradientin, eli virhefunktion osittaisderivaatat painojen ja vakiotermien suhteen laskeminen käy kätevästi kaavoilla
```math
\frac{\partial C_X}{\partial w^{(l)}_{jk}}=\delta^{(l)}_ja^{(l+1)}_k\quad\text{ja}\quad
\frac{\partial C_X}{\partial b^{(l)}_{jk}}=\delta^{(l)}_j,
```
missä $` C_X `$ on virhefunktio syötteellä $` X `$, $` w^{(l)}_{jk} `$ on tason $` l `$ paino neuronista $` k `$ neuroniin $` j `$ ja $`a^{(l+1)}_k `$ on
tason $` l + 1 `$ neuronin $` k `$ aktivaatioarvo (syötteellä $` X `$). Lisäksi
```math
\delta^{(l)}_j=\sum_k\delta^{(l+1)}_kw^{(l+1)}_{kj}a^{(l)}_j(1-a^{(l)}_j),
```
olettaen, että aktivointifunktiona on käytössä sigmoid-funktio. Viimeisen (vastavirta-algoritmin kannalta ensimmäisen) kerroksen $` \delta^{(u)} `$ saadaan
```math
\delta^{(u)}_j=2\left(\hat{a}^{(u)}_j-a^{(u)}_j\right)a^{(u)}_j(1-a^{(u)}_j),
```
missä $` \hat{a}^{(u)}_j `$ on ulostulotason $` u `$ neuronin $` j `$ toivottu tulos olettaen, että käytössä on neliövirhe
```math
C_X=\|\hat{a}-a^{(u)}\|^2.
```
Neuroverkossani nämä oletukset tulevat pitämään vähintään aluksi.

Määrittelydokumentissa mainitsemieni lähteiden lisäksi käytin [Wikipediassa](https://en.wikipedia.org/wiki/Weight_initialization) annettua neuroverkon painojen alustusmenetelmää.

### Mikä jäi epäselväksi tai tuottanut vaikeuksia?

### Mitä teen seuraavaksi?

### Ajankäyttö
Olen käyttänyt projektiin tällä viikolla noin 6 tuntia.
