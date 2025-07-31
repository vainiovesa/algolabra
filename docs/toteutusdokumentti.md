## Toteutusdokumentti

### Ohjelman yleisrakenne
Ohjelma koostuu neuroverkko-, datankäsittely- ja käyttöliittymämoduuleista.

Projektin ydin sijaitsee neuroverkkomoduulissa. Neuroverkko on funktio (tässä tapauksessa $` f:\;\mathbb{R}^{784}\rightarrow\mathbb{R}^{10} `$) missä lähtöjoukon alkiot ovat MNIST-tietokannan 28 x 28 kuvia koottuna sarakevektoreiksi. Maalijoukon alkiot kuvaavat todennäköisyyksiä, mihin luokkaan annettu kuva kuuluu (mikä numero nollasta yhdeksään kuvassa on). Esimerkiksi vektori $` [0\;1\;0\;0\;0\;0\;0\;0\;0\;0]^T `$ luokittelee annetun kuvan ykköseksi*. Neuroverkko on itse asiassa yhdistetty funktio, jossa syöte kulkee kerrosten läpi. Yksittäinen kerros lasketaan $` \sigma(Wx+b) `$, missä $` x `$ on syöte, $` W `$ on painomatriisi, $` b `$ on vakiotermivektori ja $` \sigma(.) `$ aktivointifunktio. Seuraava kerros on edellisen ulkofunktio. Tässä projektissa neuroverkon aktivointifunktiot ovat kaikki sigmoid-funktioita.

Neuroverkkoa koulutetaan, eli sen painoja ja vakiotermejä säädellään eri gradienttimenetelmillä (perinteinen, stokastinen ja minisatsi), joiden ero on lähinnä se, kuinka usein parametreja päivitetään. Yksinkertaisuudessaan jokaisella koulutusesimerkillä (kuva ja sen luokka) lasketaan tappiofunktion gradientti neuroverkon parametrien suhteen ja parametrit päivitetään pienentämään tappiofunktion arvoa. Tässä projektissa tappiofunktiona käytetään neliövirhettä. 

Käyttöliittymässä käyttäjä voi luoda uuden neuroverkon, kouluttaa ja testata sitä, sekä tallentaa verkon. Käyttäjä voi myös ladata aiemmin tallentamansa verkon. Vaikka neuroverkko on periaatteessa mielivaltaisen kokoinen, on käyttäjälle tietyt rajat muistin säästämiseksi. Koulutusvaiheessa käyttäjä saa päättää, millä gradienttimenetelmällä neuroverkkoa koulutetaan ja antaa epookkien määrän, sekä oppimisnopeuden. Minisatsigradienttimenetelmän tapauksessa käyttäjä määrittelee myös minisatsin koon. Koulutuksen valmistuttua käyttäjälle näytetään graafi tappiofunktion arvon ja vahvistusdatan luokittelun kulusta koulutuksen aikana. Neuroverkkoa testatessa käyttäjälle näytetään neuroverkon luokittelutarkkuus testidatalla. Käyttäjä näkee myös neuroverkon oikein ja väärin luokittelemia kuvia.

Datankäsittelymoduuli lataa MNIST-tietokannan kuvat ja muuttaa ne neuroverkolle sopivaan muotoon.

*: Tällainen vektori saadaan vasta luokitteluvaiheessa. Neuroverkon antama vektori ei ole välttämättä (eikä yleensä) yksikkövektori.

### Saavutetut aika- ja tilavaativuudet 
Neuroverkko ja sen syöte pysyy aina samankokoisena, joten sen tila ja nopeus pysyvät vakioina.

### Työn mahdolliset puutteet ja parannusehdotukset
Neuroverkko käyttää vain sigmoid-aktivointifunktiota ja neliövirhe-tappiofunktiota.

### Laajojen kielimallien käyttö
Tässä projektissa ei ole käytetty laajoja kielimalleja.

### Käytetyt lähteet
- [First Principles of Computer Vision](https://www.youtube.com/watch?v=sIX_9n-1UbM)
- [Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Heli Tuominen](https://tim.jyu.fi/view/143092#virhe)
- [Sebastian Björkqvist](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)
