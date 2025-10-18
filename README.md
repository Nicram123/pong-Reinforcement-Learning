# Pong Reinforcement Learning (Prawa AI , lewa paletka gracz/user)
## Opis projektu
Ten projekt to implementacja gry Pong z wykorzystaniem uczenia ze wzmocnieniem (Reinforcement Learning). Celem projektu jest nauczenie prawej paletki reagować na odbijanie piłki przez usera (lewa strona)
## Instalacja i uruchomienie
1. Sklonuj repozytorium: `git clone`
```bash
git clone git@github.com:Nicram123/pong-Reinforcement-Learning.git
```                                                                      
3. Zainstaluj wymagane biblioteki:
```bash
pip install pygame
```
5. Uruchom trening:
```bash
python -m train
```
lub skorzystaj z gotowych modeli w folderze `models`

7. Uruchom program z poziomu `main.py`: 
```bash
python main.py
```
## Trening
Trening Paletki odbywa się tak że:
* jeden epizod trwa dopóki ktoś nie zdobędzie punktu, podczas trenowania lewa strona (user) jest symulowana po przez losowe ruchy w stronę piłki zmierzającej do niego
*  na podstawie symulowanego lewego gracza , AI (prawa strona uczy się ruchu, żeby gra trwała jak najdłużej)
## Wyniki po treningu 
![pong](https://github.com/user-attachments/assets/92e00ec3-4eed-4e49-b11e-3934392b79a1)
## Uwagi
* Na razie najlepszy z modeli `pong_ai_epq600.keras` z folderu `models`




