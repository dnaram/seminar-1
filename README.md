# Seminar: Segmentacija medicinskih slika jetre pomoću Monaija i PyTorcha

Projekt *Seminar: Segmentacija medicinskih slika jetre pomoću Monaija i PyTorcha* bavi se označavanjem dijelova slike koja predstavlja medicinski nalaz i na kojoj se nalazi jetra kao ciljni objekt označavanja. Cilj ovog projekta je prikazati arhitekturu postojećih modela dubokog učenja koji se koriste za segmentiranje objekata sa slika medicinskih nalaza, demonstrirati njihov rad korištenjem programskog jezika Python te odgovarajućih programskih biblioteka te ispitati performanse takvih modela i njihovu točnost. Osim za demonstraciju rada postojećih modela, ovaj projekt služi i kao tehnički uvod sadržavajući ulomke koji prikazuju postupke pripreme skupa podataka, njihova pretprocesiranja, ali i instalaciju potrebnih paketa te česte probleme i pogreške na koje je moguće naići prilikom uporabe istih.

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
Budući da se za daljnje potrebe projekta očekuje da se korišteni podatci dostavljaju u dogovorenom obliku, potrebno ih je pripremiti i predobraditi. S obzirom na to da će se određeni dio tih dvaju procesa obaviti automatizirano, dok će se dio obraditi manualno - napravljena je finija podjela na poglavlja koja se bave pripremom i pretprocesiranjem podataka. 

Na sljedećoj [poveznici](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) nalaze se medicinske slike jetre kao i ispravne labele korištene u daljnjem dijelu rada. Preuzete slike dimenzijski su određene svojom visinom, širinom, ali i brojem razina (engl. *slices*) koje predstavljaju svojevrsni horizontalni presjek organa koji se uzimaju u razmatranje, a koji, naravno, imaju i prostorno zauzeće. Problem će predstavljati činjenica da broj razina za svakog pacijenta (sliku) nije jednak. 

Obrada podataka uključuje pretvorbu nifti datoteka u dicom datoteke. Za taj postupak koristi se alat 3D Slicer i njegova opcija koju nudi **Create a Dicom Series**. Na ovaj je način moguće i za svakog pacijenta manualno postaviti broj razina od interesa i riješiti se ranije uvedenog problema.  Međutim, kako bi se izbjegla gnjavaža manualnog postavljanja broja razina za svakog pacijenta - napisana je skripta `preprocess.py` koja automatizira navedeni posao. Ujedno u ovoj se skripti nalazi i funkcija koja radi pretvorbu između nifti i dicom datoteka. 

## 3. Pretprocesiranje
Za izradu ovog dijela projetka koristit će se *open source* alat monai, koji se temelji na PyTorchu. Prvi korak je instaliranje biblioteke pomoću `pip` ili `conda`, ovisno o okruženju koje se koristi. Preporuka je postaviti virtualno okruženje za projekt jer ova biblioteka ne radi uvijek kada je instalirana izravno u sustav.

Kako bi se primijenilo više transformacija na istog pacijenta, koristi se monaijevu funkciju `compose`, koja  omogućuje kombiniranje bilo koje transformacije po izboru (one definirane u monai dokumentaciji).

Prilikom korištenja monaija, primarne transformacije su `Load image` za učitavanje *nifty* datoteka i `ToTensor` za pretvaranje transformiranih podataka u *torch tenzore* kako bi ih se moglo koristiti za treniranje.

Nakon navedenih temeljnih transformacije, vrijedno je istaknuti i one koje se pokazuju korisnima u praksi: `AddChanneld`, `Spacingd`, `ScalIntensityRanged`, `CropForegroundd`, `Resized`.

## 4. Instalacija potrebnih paketa
Priprema radne okoline za pokretanje programskog koda uključuje instalaciju programskog jezika Python (budući da se koriste programske biblioteke Monai i PyTorch), preferiranu razvojnu okolinu (npr. VSCode, PyCharm), 3D slicer za prikazivanje podataka te ITK snap korekciju segmentacija (možda neće biti korištenu u ovom projektu, ali generalno je koristan alat za probleme segmentiranja). 

Instalacija programskog jezika Python uobičajeno započinje odlaskom na službene stranice[^5^](#5) te klikom na gumb **Download** za instalaciju najnovije verzije (ili neke specifične) za odgovarajući operacijski sustav. Nakon odabira željene putanje u kojem će se pohraniti potrebne datoteke započinje preuzimanje čije trajanje ovisi o kvaliteti internet veze. Nakon toga potrebno je provesti inicijalni setup kojeg je moguće pratiti s postojećih Youtube tutorijala[^6^](#6).    

U ovom projektu koristi se razvojna okolina Visual Studio Code[^7^](#7), međutim moguće je koristiti bilo koju drugu razvojnu okolinu pri čijem se korištenju programer osjeća ugodnije. Za potrebe ovog projekta preuzeta je verzija za operacijski sustav Windows te napravljen inicijalni setup potvrdom svih pretpostavljenih postavki.

3D Slicer[^8^](#8) je besplatni i otvoreni programski paket za analizu slika i znanstvenu vizualizaciju. U trenutku pisanja ovog dokumenta preuzeta je verzija 4.11.20210226 za operacijski sustav Windows.

ITK-SNAP[^9^](#9) je softverska aplikacija koja se koristi za segmentiranje struktura u 3D medicinskim slikama. Instalacija navedenog programa svodi se na odabir prikladne verzije (s obzirom na operacijski sustav) te ispunjavanjem podataka koji služe tvorcima aplikacije (moguće i preskočiti).

## 5. Česti problemi i pogreške
Praćenjem ovog dokumenta u želji za reproduciranjem rezultata seminara moguće je napraviti pogreške koje će kasnije rezultirati neispravnim rezultatima ili pogreškama koje konzumiraju vrijeme za njihovo ispravljanje. 

Pogrešno navedena putanja do medicinskih slika jetre ili pripadajućih labela jedna je od mogućih pogrešaka koju je vrlo lako napraviti pogotovo kad je riječ o tipfeleru u duljim nazivima datoteka i foldera. Ovakve pogreška može rezultirati greškom `TypeError` u kojoj je `DataLoader` prazan. 

Pogrešna vrijednost ključa u rječniku rezultirat će greškom tipa `KeyError` iz čije će se poruke dati zaključiti o kakvom se konkretno propustu radi. Zbog ovakve pogreške tranformacije koje radi monai neće raditi ispravno (ako i uopće).

## 6. Treniranje modela
Skripta napisana za ovaj dio projekta sadržana je u `train.py`.  Metrika koja se bavi analizom performansi treniranja i testiranja modela sadržana je u skripti `utilities.py`. 

Konkretan model predstavlja ranije opisana poznata mreža **UNet** importana iz modula `monai.networks.nets`. Vrijednost parametra `dimensions` postavljena je na `3` što je prirodan odabir budući da se radi o segmetaciji ogana jetra koji ima svoje prostorno zauzeće. Vrijednost parametra `in_channel` postavljena je na `1` budući da se radi s maskom koja ima samo jedan kanal, no vrijednost parametra `out_channel` postavljena je na `2` budući da je bitno razdvojiti pozadinu od onoga što je ispred nje. 

## 7. Testiranje modela
Skripta koja se bavi ovim dijelom projekta napisana je u `testing.ipynb`. Iz skripte je vidljivo da je nad istreniranim modelom računat gubitak *dice loss* za kojeg je iz nacrtanih grafova vidljivo da je nakon određenog broja epoha porast epoha bio irelevantan za smanjenje gubitka budući da je model došao do platoa na skupu podataka za testiranje. Također, moguće je vidjeti kako model konkretno radi nad zadanim slikama jetre te segmentira dijelove koji joj pripadaju od pozadine.

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
