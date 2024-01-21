# BMEVIMIAC10 Mesterséges intelligencia 1. házi feladat

A döntési fák (decision tree) talán az emberi gondolkodáshoz legközelebb álló, intuitív osztályozók
(classifier). Általában relatíve kevés adattal is hatékonyan taníthatóak, és a struktúrájuk emberi
szemmel is érthető. Egyszerűségük ellenére is igen jó előrejelző teljesítményt nyújtanak, így sikeresen alkalmazhatóak különböző területeken az orvosi döntéstámogatástól kezdve, az üzleti életen át
az ajánló rendszerekig (recommender systems).

## Feladat

Implementáljon Python programozási nyelven egy döntési fát. A
megoldását egyetlen, solution.py állományként várjuk a Moodle rendszerében.
A döntési fa célja az lesz, hogy egy publikus adathalmazon, a California Housing Prices2 adathalmazon végzett tanulás után eldöntse, hogy egy adott tulajdonságokkal rendelkező házat megvenne-e
egy vásárló vagy sem. (A vásárló döntését véletlenszerűen sorsoljuk a házi feladat ellenőrzése
közben.3
) Minden adatot egész számokra kerekítettünk, így az előzőekben leírt algoritmussal létre
lehet hozni egy döntési fát.
A feladat megoldásához a Moodle rendszerben elérhet néhány állományt:

1. solution.py: a megoldás minimális vázát tartalmazó állomány. A házi feladat megoldása
során célszerű ezt az állományt bővíteni.

3. train.csv: Tanító adathalmaz. A szokásos csv formátumnak megfelelő állomány (táblázatos
adatokat tárolunk olymódon, hogy egy-egy rekordot az állomány egy-egy sora reprezentál, az
adattagok pedig , (vessző) karakterrel vannak elválasztva). A feldolgozást segítendő az állományban nem szerepelnek fejlécek4
, így az első kiolvasható sor az első rekordot tartalmazza.
Az utolsó oszlopban szerepel a döntés értéke, mely 0, ha a döntési hamis, azaz nem vásárolja
meg a képzeletbeli vásárló az adott paraméterekkel bíró házat, illetve a döntés értéke 1, ha a
megvásárlás mellett dönt.

5. test.csv: A tanító adathalmazhoz hasonló felépítésű adatsor azzal a különbséggel, hogy ezen
állomány nem tartalmaz döntési oszlopot.
