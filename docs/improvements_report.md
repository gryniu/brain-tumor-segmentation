# Raport: przegląd kodu i usprawnienia (2026-11-02)

## Punkt wyjścia

- Przeczytałam `src/brain_seg/*` (dane, transformacje, model, funkcje strat, trening, inferencja, metryki), skrypty i testy.
- Kod był już solidny: podział na poziomie pacjenta z weryfikacją wycieków, atomowy zapis checkpointów,
  `weights_only=True`, odwracanie preprocessingu przy predykcji, jawne konwencje dla pustych masek w metrykach.
  **Nie znalazłam błędów poprawności w logice treningu, strat ani metryk**, więc ich nie zmieniałam.
- Prawdziwe ograniczenie wyszło z Twoich pomiarów na Colabie T4 (2 CPU): GPU czekało na dane
  (1,51 s z 2,19 s na krok), epoka ~32 min, a bufor `disk` zajmuje ~160 MB na przypadek (pilot: 8,4 GB
  na ok. 50 przypadków), więc cały zbiór nie mieścił się na dysku. Przy budżecie ok. 1 godziny to było
  decydujące, więc na tym skupiły się zmiany.
- Testy przed zmianami: 55 przeszło. Po zmianach: wszystkie przechodzą (patrz niżej).

## Co zmieniłam

1. **Kompaktowy bufor `data.cache: npy`** (`src/brain_seg/data.py`, klasa `CompactCacheDataset`).
   Obraz zapisywany jako float16, maska jako uint8, odczyt przez memory-map. Zapis atomowy (bezpieczny dla
   wielu workerów), uszkodzone lub ucięte pliki są odbudowywane automatycznie.
   - Zmierzone na pełnowymiarowych **fantomach** (240×240×155, na Macu): **47 MB na przypadek zamiast 179 MB**
     w trybie `disk` (3,8× mniej), a czas odczytu bez zmian (ok. 0,05-0,06 s na przypadek po zbudowaniu
     bufora, wobec 1,0 s bez bufora).
   - To są pomiary na fantomach, nie na prawdziwym MRI i nie na Colabie. Dla prawdziwych danych szacuję
     (ekstrapolacja z pomiaru pilota: 162 MB/przypadek w `disk`) rzędu 40-45 MB na przypadek, czyli ~38 GB
     dla 874 przypadków treningowych. To nie jest pomiar.
   - Maska jest zapisywana bezstratnie. Obraz traci precyzję do względnego błędu ok. 5·10⁻⁴ (test:
     `rtol=2e-3`), co jest poniżej szumu augmentacji (std 0,1).
2. **Budżet czasu `train.max_hours` działa teraz na wprost** (`src/brain_seg/train.py`).
   Wcześniej pętla sprawdzała limit dopiero po zakończeniu epoki, więc przy epoce 32 min i limicie 1 h mogła
   przekroczyć budżet o pół godziny. Teraz nie zaczyna epoki, która by się nie zmieściła (zakłada czas równy
   ostatniej epoce), a przed zatrzymaniem zawsze robi walidację, żeby `best.pt` odpowiadał ostatnim wagom.
   Wcześniej ostatnia epoka mogła zostać niezwalidowana.
3. **`data.cache_dir`** (`src/brain_seg/config.py`, `train.py`): bufor można trzymać na szybkim lokalnym
   dysku, a wyniki (checkpointy) na Dysku Google. Wcześniej bufor zawsze lądował w `output_dir/cache`, czyli
   na Dysku, który ma tylko 11 GB wolnego i jest wolny.
4. **`configs/colab_t4_1h.yaml`**: konfiguracja na ok. godzinę na T4 (bufor `npy`, mniejszy model 5,65 mln
   parametrów, patch 96³, 250 przypadków treningowych, walidacja na 20 przypadkach, limit ok. 39 min).
   **Wartości są planistyczne i nie były mierzone na T4.**
5. **Testy** (`tests/test_cache_and_budget.py`, 8 nowych; łącznie 64 przechodzą, bo nowy plik konfiguracji jest też sprawdzany istniejącym testem): zgodność bufora z potokiem bez bufora (maska
   identyczna, obraz w tolerancji), budowa bufora tylko raz, rozmiar, odbudowa uszkodzonego pliku, kształty
   próbek treningowych, trening end-to-end z buforem, limit czasu (zatrzymanie plus końcowa walidacja) i brak
   wpływu limitu przy dużym budżecie.
6. **Refaktoryzacja bez zmiany działania**: losowa część potoku treningowego wydzielona do
   `sampling_and_augmentation_transforms` (`src/brain_seg/transforms.py`). Dodano rzutowanie obrazu na float32
   po wycięciu patcha (dla float32 to operacja pusta). README zaktualizowany.

## Czego nie zrobiłam i dlaczego

- Nie zmieniałam modelu, strat, optymalizatora ani augmentacji: nie mam jak zmierzyć na GPU, czy by pomogły,
  a nie chcę zgadywać na kodzie, który działa.
- Nie testowałam niczego na GPU ani na prawdziwych danych BraTS z tego komputera (nie ma ich tu). Wszystkie
  testy używają fantomów i mówią tylko, że kod jest spójny, nie jak dobry będzie model.
- `limit_train_cases` bierze pierwsze N przypadków z listy treningowej, a nie losową próbkę. Nie zmieniałam
  tego, żeby nie unieważnić wcześniejszych uruchomień (pilot użył pierwszych 40).
- Szybkość na Colabie T4 jest nieznana, dopóki nie uruchomisz treningu.

## Wyniki modelu

Brak. Jedyne liczby jakości to metryki walidacyjne z pilota (40 przypadków, 12 epok, walidacja na 10
przypadkach): średni Dice 0,601 w epoce 12. To kontrola poprawności, nie wynik projektu.

---

# Część 2: zestaw usprawnień przed pełnym treningiem (2026-11-02)

Zaimplementowane i przetestowane jednostkowo (86 testów przechodzi, w tym 18 nowych w `tests/test_improvements.py` i 4 sprawdzenia nowych plików konfiguracji).
**Nic z tego nie było jeszcze trenowane ani mierzone na prawdziwych danych ani na GPU.** Wynik bazowy w README
(Dice ET 0,825 / TC 0,871 / WT 0,903) pochodzi z modelu bez żadnej z tych zmian.

| # | Pomysł | Co zrobiłam | Gdzie |
|---|---|---|---|
| 1 | Więcej danych i epok | Konfiguracja na wszystkie 874 przypadki, 60 przypadków do wyboru modelu, limit czasu | `configs/colab_t4_full.yaml` |
| 2 | Większy patch i model | Patch 128³, 16,6 mln parametrów | `configs/colab_t4_large.yaml` |
| 3 | Post-processing | Usuwanie małych składowych, mały ET przeklasyfikowany na NCR, skrypt do strojenia progów **na walidacji** (zbiór testowy jest odrzucany) | `src/brain_seg/postprocess.py`, `tuning.py`, `scripts/tune_postprocessing.py` |
| 4 | TTA | Uśrednianie logitów po odbiciach (`flips3` lub `flips8`), tylko w evaluate/predict | `inference.tta` w `src/brain_seg/inference.py` |
| 5 | Trening na regionach | 3 wyjścia sigmoid (ET/TC/WT), strata Dice + BCE, dekodowanie z progiem 0,5 i zagnieżdżaniem | `model.output: regions` (`labels.py`, `losses.py`, `model.py`, `train.py`, `inference.py`) |
| 6 | Analiza porażek | Klasyfikacja porażek (pominięty, za mały, za duży, w złym miejscu, fałszywy alarm), zależność Dice od objętości, 10 najgorszych przypadków; tylko liczby, bez obrazów pacjentów | `scripts/analyze_failures.py`, `src/brain_seg/analysis.py` |
| 7 | Rzetelność oceny | 95% przedziały ufności z bootstrapu **po pacjentach** (skany tego samego pacjenta są skorelowane), porównanie parami dwóch przebiegów | `summary.md`, `scripts/compare_eval.py` |

Dodatkowo: `evaluate.py` i `predict.py` przyjmują `--set inference.*=...`, a użyte ustawienia zapisują się w podsumowaniu.
Domyślne wartości nowych opcji niczego nie zmieniają, więc stare konfiguracje i checkpointy działają jak wcześniej.

## Weryfikacja

* Wszystkie 86 testów przechodzi. Test TTA wykrywa błąd: po usunięciu odwracania odbicia dwa testy TTA przestają przechodzić
  (sprawdzone, potem cofnięte).
* Dekodowanie regionów jest odwracalne dla każdej mapy etykiet, a niespójne predykcje (ET bez TC itp.) dają poprawne,
  zagnieżdżone etykiety.
* Metryki na przyciętym obszarze w strojeniu post-processingu są takie same jak na całym wolumenie.

## Czego nie wiem

* Czy którakolwiek z tych zmian poprawia model na prawdziwych danych. Trzeba to zmierzyć na **walidacji**, jedna zmiana naraz.
* Ile czasu i miejsca na dysku zajmą pełne dane i większy model na T4. Liczby w nowych konfiguracjach to szacunki z poprzednich
  pomiarów (78 s na epokę dla 250 przypadków, 4,78 GB pamięci GPU dla 128³), nie pomiary.
* Czy trening na regionach jest lepszy niż softmax w tym projekcie: w nnU-Net jest standardem, ale tu nie jest sprawdzony.
