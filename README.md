# Athlete Tracker

Das vorliegende Script ermöglicht das Tracken eines Athleten.
Die Hilfe zum Script kann über `athlete-tracker.exe --help` aufgerufen werden.

Bei aktivierter automatischer Objekterkennung wird nach dem Einlesen des ersten Frames die Objekterkennung initialisiert. Als Detektor kommt ein Neuronales Netz vom Typ MobileNetSSD zum Einsatz. Ist die automatische Erkennung nicht aktiviert, kann der Benutzer den Bereich (region of interest) selbst wählen.

Ist die Erkennung erfolgreich, wird das Ergebnis an den Tracker übergeben. Dieser wird anschließend mit jedem Frame aktualisiert und passt die Bounding Box an die neue Position des Athleten im Bild an.

Die Ausgabe der Box-Koordinaten erfolgt im Textformat direkt an die Konsole.
