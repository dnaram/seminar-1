# Seminar: Segmentacija jetre pomoću Monaija i PyTorcha

Projekt *Seminar: Segmentacija jetre pomoću Monaija i PyTorcha* bavi se označavanjem dijelova slike koja predstavlja medicinski nalaz i na kojoj se nalazi jetra kao ciljni objekt označavanja. Cilj ovog projekta je prikazati arhitekturu postojećih modela dubokog učenja koji se koriste za segmentiranje objekata sa slika medicinskih nalaza, demonstrirati njihov rad korištenjem programskog jezika Python te odgovarajućih programskih biblioteka te ispitati performanse takvih modela i njihovu točnost. Osim za demonstraciju rada postojećih modela, ovaj projekt služi i kao tehnički uvod sadržavajući ulomke koji prikazuju postupke pripreme skupa podataka, njihova pretprocesiranja, ali i instalaciju potrebnih paketa te česte probleme i pogreške na koje je moguće naići prilikom uporabe istih.

## 1. U-Net: definicija arhitekture

Prije demonstracije segmentiranja izvođenjem programskog koda, namjera je ukratko prikazati arhitekturu mreže U-Net - konvolucijske mreže za segmentaciju biomedicinskih slika[^1^](#1). U-Net na ulazu dobiva biomedicinsku sliku nakon čije obrade na izlaz vraća polariziranu sliku na kojoj su pikselima s vrijednošću 1 označeni dijelovi slike za koje model smatra da pripadaju ciljnim objektima, a vrijednošću 0 prikazana je pozadina koja ne pripada ciljnom objektu.

Specifičnost ovako definiranog modela je u činjenici što, za razliku od brojnih modela dubokog učenja, što ujedno predstavlja i bitnu prepreku za njihovu općenitu uporabu, ne zahtijeva veliki broj primjera za učenje kako bi u konačnici uspješno odrađivao posao segmentiranja na neviđenim primjerima. Konvolucijske mreže, kakva je i U-net, tipično se koriste za rješavanje problema klasifikacije u kojima je izlaz takvih mreža uobičajeno oznaka (engl. *label*) kojom model predviđa pripadanje ulazne slike određenoj klasi iz ranije definiranog skupa. Međutim, u brojnim problemima iz područja računalnog vida, a posebno u području biomedicinske obrade slika, pojavljuje se potreba u kojoj bi očekivani izlaz trebao uključivati lokalizaciju (engl. *localization*), tj. oznaka klase bi trebala biti dodijeljena svakom pikselu sa slike. Nadalje, ozbiljan napad na upotrebljivost brojnih modela dubokog učenja jest česta nemogućnost za pristup tisućama biomedicinskih slika koje bi poslužile kao podatci za učenje.

Arhitektura mreže U-Net sastoji se od sažimajućeg (engl. *contracting*) dijela za detektciju konteksta i simetričnog proširujućeg (engl. *expanding*) dijela koji omogućuje preciznu lokalizaciju. Sam naziv mreža je dobila zahvaljujući upravo svojoj strukturi koja svojim oblikom podsjeća na oblik slova "U". Prije predlaganja mreže U-Net za probleme segmentiranja, najboljom metodom pokazivala se konvolucijska mreža kliznog prozora (engl. *sliding window convolutional network*). Primjer takve mreže korišten je u radu Dana Ciresana i njegovih kolega[^2^](#2) u kojem se klasa pojedinog piksela pokušala odrediti na temelju vrijednosti piksela koji predstavljaju lokalno okruženje (engl. *patch*) na kojeg možemo gledati kao prozor oko tog piksela - otuda i naziv metode. Pozitivne strane ovakvog pristupa jesu činjenica da model na ovakav način može lokalizirati te da se ovim pristupom skup podataka uvećava s obzirom na početnu količinu slika korištenih za učenje modela. Međutim, nedostatci ovakvog pristupa manifestiraju se u sporosti - budući da je mrežu potrebno pokretati za svako lokalizirano područje, a budući da se područja umnogome preklapaju, dolazi do redundancije. Nadalje, javlja se kompromis između točnosti lokalizacije i korištenja konteksta. Veći prozori zahtijevaju više max-pooling slojeva[^3^](#3) koji smanjuju točnost lokalizacije, dok manji prozori daju mreži na raspolaganje manje konteksta.  

Pristup na kojem se zasniva mreža U-Net nastoji riješiti prethodno navedene probleme. Glavna ideja je nadopuniti uobičajeni sažimajući (engl. *contracting*) dio mreže
uzastopnim slojevima, gdje se operatori skupljanja (engl. *pooling operators)* zamjenjuju operatorima povećanja uzorkovanja (engl. *upsampling operators*).
Stoga ovi slojevi povećavaju rezoluciju izlaza modela. Kako bi se zadovoljio zahtjev lokalizacije, značajke visoke razlučivosti s sažimajućeg dijela kombiniraju se s povećanim uzorkovanjem. Uzastopni konvolucijski sloj tada može naučiti sastaviti precizniji
izlaz na temelju ovako dobivenih informacija.
 
![](https://raw.githubusercontent.com/nibtehaz/MultiResUNet/master/imgs/unet.png) 
**Slika 1.** Prikaz arhitekture mreže U-Net

Arhitektura mreže prikazana je na **Slika 1.**. Sastoji se od sažimajućeg dijela (lijeva strana mreže) i proširujućeg dijela (desna strana mreže). Sažimajući dio oslanja se na uobičajenu strukturu konvolucijskih mreža. Sastoji se od ponovljene primjene dviju konvolucijskih mreža 3x3, nakon kojih slijedi izlaz zglobnice (engl. *Rectified Linear Unit, ReLU*) te 2x2 max-pooling operacija s korakom (engl. *stride*) postavljenim na vrijednost 2 za smanjenje uzorkovanja (engl. *downsampling*). U svakom koraku smanjenja uzorkovanja broj kanala značajki se udvostručuje. Proširujući dio sastoji se od (simetrično) povećanja uzorkovanja (engl. *upsampling*) značajki nakon kojeg slijedi konvolucija 2x2 koja prepolovljava broj kanala značajki, konkatenacije s prethodno izrezanim značajkama iz sažimajućeg dijela te dvije 3x3 konvolucije nakon kojih slijedi izlaz zglobnice.


## 2. Priprema skupa podataka


## 3. Pretprocesiranje

## 4. Instalacija potrebnih paketa
Priprema radne okoline za pokretanje programskog koda uključuje instalaciju programskog jezika Python (budući da se koriste programske biblioteke Monai i PyTorch), preferiranu razvojnu okolinu (npr. VSCode, PyCharm), 3D slicer za prikazivanje podataka te ITK snap korekciju segmentacija (možda neće biti korištenu u ovom projektu, ali generalno je koristan alat za probleme segmentiranja). 

Instalacija programskog jezika Python uobičajeno započinje odlaskom na službene stranice[^5^](#5) te klikom na gumb **Download** za instalaciju najnovije verzije (ili neke specifične) za odgovarajući operacijski sustav. Nakon odabira željene putanje u kojem će se pohraniti potrebne datoteke započinje preuzimanje čije trajanje ovisi o kvaliteti internet veze. Nakon toga potrebno je provesti inicijalni setup kojeg je moguće pratiti s postojećih Youtube tutorijala[^6^](#6).    

U ovom projektu koristi se razvojna okolina Visual Studio Code[^7^](#7), međutim moguće je koristiti bilo koju drugu razvojnu okolinu pri čijem se korištenju programer osjeća ugodnije. Za potrebe ovog projekta preuzeta je verzija za operacijski sustav Windows te napravljen inicijalni setup potvrdom svih pretpostavljenih postavki.

3D Slicer[^8^](#8) je besplatni i otvoreni programski paket za analizu slika i znanstvenu vizualizaciju. U trenutku pisanja ovog dokumenta preuzeta je verzija 4.11.20210226 za operacijski sustav Windows.

ITK-SNAP[^9^](#9) je softverska aplikacija koja se koristi za segmentiranje struktura u 3D medicinskim slikama. Instalacija navedenog programa svodi se na odabir prikladne verzije (s obzirom na operacijski sustav) te ispunjavanjem podataka koji služe tvorcima aplikacije (moguće i preskočiti).

## 5. Česti problemi i pogreške

## 6. Treniranje/učenje modela

## 7. Testiranje modela

## 8. Izvori

1. <a id="1"></a>https://link.springer.com/content/pdf/10.1007%2F978-3-319-24574-4_28.pdf
2. <a id="2"></a>Ciresan, D.C., Gambardella, L.M., Giusti, A., Schmidhuber, J.: Deep neural networks segment neuronal membranes in electron microscopy images. In: NIPS, pp. 2852–2860 (2012)
3. <a id="3"></a>https://paperswithcode.com/method/max-pooling
4. <a id="4"></a>https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
5. <a id="5"></a>https://www.python.org/downloads/
6. <a id="6"></a>https://www.youtube.com/watch?v=uDbDIhR76H4&t=2s
7. <a id="7"></a>https://code.visualstudio.com/download
8. <a id="8"></a>https://download.slicer.org/
9. <a id="8"></a>http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3