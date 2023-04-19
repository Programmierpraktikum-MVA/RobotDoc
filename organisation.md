# RobotDoc

:::info
Automatische Diagnose – a) Identifikation von Mustern, b) automatische Anerkennung von Krankheiten und c) Vorschlag von Behandlung. [^1]
:::

> **Note:** das Dokument dient aktuell nur als Entwurf und Grundlage zur Organisation (da großes Team). Ich habe ein paar vorläufige Ideen skizziert. Gerne Feedback via Kommentar! - [Noel](https://wa.me/4915678381262)

## Ressourcen

- Kommunikation:
    - [Discord](https://discord.gg/Ut2wyECa) (main)
    - [WhatsApp](https://chat.whatsapp.com/DuXWToRfYMFI7cbBX6HDQG) (chat)
    - [Matrix](https://matrix.to/#/!ANIUCMrXTsCJbzRwmC:matrix.tu-berlin.de?via=matrix.tu-berlin.de)
- [GitHub Repository](https://github.com/Programmierpraktikum-MVA/RobotDoc) 
- [ISIS](https://isis.tu-berlin.de/course/view.php?id=33313#section-0)
- [Cloud](https://drive.google.com/drive/folders/1hdzV838ZeBf8juitnRhTfD-ElWRsil52)

## Up Next

- [ ] Code aus [Vorjahren](https://drive.google.com/drive/folders/1OKv1ZZOOsZrgzUp5hKcEM74_CDabGl5s) ansehen [bis Montag 24.04.]
- [ ] Projekt in [Module](#Module) aufteilen 
- [ ] [Personen](#Team) zu [Modulen](#Module) zuordnen 
- [ ] Ideen skizzieren (für [Erstes Design](#Termine))

## Termine

* **Ablaufrücksprache:** jeden Montag (16:10 - 16:35)
* ~~**Kick-Off:** 17.04. (16:00 - 18:00)~~
* **Erstes Design:** 01.05. (16:00 - 18:00)
* **Milestone 1:** 22.05. (16:00 - 18:00)
* **Milestone 2:** 12.06. (16:00 - 18:00 Uhr)
* **Milestone 3:** 03.07. (16:00 - 18:00 Uhr)
* **Belastungstest:** 10.07. (16:00 - 18:00 Uhr)
* **Unterlagenabgabe:** 31.07. (15:59 Uhr) [^2]

## Module

0. **Architektur** (2 Personen)
0.1. Struktur und Organisation
0.2. Schnittstellen (definieren und sicherstellen)

1. **Daten** (3 Personen)
1.1. Aufbereitung
1.2. Datenbank

2. **Machine Learning** (8 Personen)
2.1. Muster / Krankheiten erkennen 
2.2. passende Behandlung suchen 

3. **Applikation** (4 Personen)
3.1. Integration
3.2. Darstellung

## Tool Stack

- Sprache: [Python](https://docs.python.org/3/) [^4]
- Datenbank: tba
- Machine Learning: [TensorFlow](https://www.tensorflow.org/learn)
- Applikation: [Flask](https://flask.palletsprojects.com/en/2.2.x/) / [Django](https://www.djangoproject.com/start/)

## Prinzipien 

- iterativ
- agil 
- modular (unabhängig mit Schnittstellen)
- lean

## Team

> **Notation:** Vorname Nachname (relevante Skills) [bevorzugte Module]

* Tamer Aktas (Frontend/Backend, Projektmanagement)
* Ulas Can Akyildiz
* Daniel Sebastian Aust
* George Badour
* David Maximilian Fränkle
* Emin Egemen Hidiroglu
* Feidi Kallel
* Irem Kement
* Annabelle Elise Kröger
* Noel Leon Kronenberg (Projektmanagement, Programmstruktur, Design, KI Basics, Charité Kontakt) [0, 2, 3]
* Jonas Arthur Krüger
* Till Friedemann Kurek
* Burak Arda Okutan
* Amor Rezgui
* Anton Noah Saatz
* Emirhan Sarioglu
* Baki Berkay Uzel [^3]

[^1]: https://isis.tu-berlin.de/course/view.php?id=33313#section-2
[^2]: https://isis.tu-berlin.de/course/view.php?id=33313#section-4
[^3]: https://isis.tu-berlin.de/mod/choicegroup/view.php?id=1532952
[^4]: Hintergrund: einfache Sprache (ermöglicht schnelle und einfache Mitarbeit), viele (ML) Bibliotheken, einfache Web-Frameworks