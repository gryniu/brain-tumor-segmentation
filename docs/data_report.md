# Raport o danych: BraTS 2023 Adult Glioma (BraTS-GLI), zbiór treningowy

Stan na 2026-11-02. Źródło wszystkich liczb: wyniki `scripts/inspect_dataset.py` i `scripts/smoke_test.py`
uruchomione w Colabie (Tesla T4) na pobranym zbiorze, wklejone przez użytkownika. Nic tu nie jest wynikiem
treningu. Czego nie zmierzono, jest oznaczone jako "nie zmierzono".

## Zbiór

| Pozycja | Wartość |
|---|---|
| Wyzwanie | ASNR-MICCAI BraTS 2023, Adult Glioma (BraTS-GLI), dane treningowe |
| Źródło | Synapse syn51514132 (archiwum treningowe), licencja CC-BY-NC |
| Kompletne przypadki | **1251** (niekompletnych: 0, nierozpoznanych plików NIfTI: 0) |
| Pacjenci | **1133**; 118 pacjentów ma więcej niż jeden przypadek/punkt czasowy (w podziale trzymani razem) |
| Pliki na przypadek | 5: `t1n`, `t1c`, `t2w`, `t2f`, `seg` (po 1251 każdego) |
| Modalności wejściowe | 4: T1, T1ce, T2, FLAIR |
| Rozmiar woluminu | (240, 240, 155) we wszystkich 1251 przypadkach |
| Rozdzielczość | 1,0 x 1,0 x 1,0 mm we wszystkich 1251 przypadkach |
| Orientacja | LPS we wszystkich 1251 przypadkach |
| Kontrola geometrii | 1251 zgodnych, 0 błędów |
| Rozmiar na dysku | nie zmierzono (przed pobraniem Colab miał 87,1 GB wolnego miejsca) |

## Etykiety (profil `brats2023`)

| Wartość | Klasa | Znaczenie |
|---|---|---|
| 0 | BG | tło |
| 1 | NCR | nieulegający wzmocnieniu rdzeń guza / martwica (NETC) |
| 2 | ED | otaczająca hiperintensywność FLAIR (obrzęk, SNFH) |
| 3 | ET | guz wzmacniający się (ET) |

Statystyki klas policzone tylko na **próbce 25 masek** (nie na całym zbiorze):

| Klasa | Obecna w | Udział w objętości guza |
|---|---|---|
| BG | 25/25 | n/d |
| NCR | 22/25 | 16,5% |
| ED | 25/25 | 64,6% |
| ET | 24/25 | 18,9% |

Wniosek z próbki: klasy są silnie niezbalansowane, a tło dominuje (w partiach treningowych ok. 91-95% voxeli).
W losowo wybranych partiach (smoke test) NCR to ok. 0,5-1,2%, ET ok. 1,1-1,2% voxeli.

## Podział danych

Podział na poziomie pacjenta (seed 42, ułamki val 0,15 / test 0,15 z `configs/default.yaml`).
Smoke test podał **874 przypadki treningowe**. Dokładnych liczb walidacji i testu nie zmierzono
(razem 377 przypadków, czyli 1251 minus 874; podział między nimi zapisuje `split_seed42.json`).

## Pomiary na Tesla T4 (15 GB), konfiguracja `cloud_cuda.yaml`

| Pozycja | Wartość |
|---|---|
| Model | unet3d, 16,55 mln parametrów |
| Patch | 128x128x128, 2 patche na krok |
| Szczytowa pamięć GPU (1 krok) | 4,78 GB |
| Czas kroku | 2,19 s, z czego 1,51 s oczekiwania na dane |
| Projekcja epoki (874 przypadki) | ok. 32 min, bez walidacji; projekcja z 17 kroków, nie gwarancja |
| Wnioski | GPU czeka na dane (2 rdzenie CPU, dekompresja `.nii.gz`); pamięci GPU jest dużo zapasu |

## Co z tego wynika

- Dane są kompletne i spójne, można na nich trenować.
- Przy ~32 min/epokę domyślne 300 epok to ok. 160 godzin, więc na darmowym Colabie nierealne bez przyspieszenia
  wczytywania danych (np. `data.cache=disk`) albo mocniejszego CPU/GPU.
- Żadnych wyników modelu jeszcze nie ma. Metryki raportujemy dopiero z `eval_test/summary.md`.
