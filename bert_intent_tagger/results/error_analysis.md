# Rule-based comparison

Both systems were evaluated against `source_corpus_exact_match_v1` recovered test boundaries.

- BERT exact intent-set accuracy: 0.997972
- Rule-based exact intent-set accuracy: 0.276783

## BERT correct, rule-based wrong
- `list california airports , list la and how many canadian airlines international flights use aircraft 320`
  - reference: [(0, 2), (4, 5), (7, 15)]; rule: [(0, 5), (7, 15)]
- `show me the cheapest round trip fares from san francisco to houston and then how many passengers can an l1011 aircraft hold`
  - reference: [(0, 11), (14, 21)]; rule: [(0, 21)]
- `what cities does northwest fly to and list the distance in miles from san francisco international airport to san francisco downtown`
  - reference: [(0, 5), (7, 20)]; rule: [(0, 20)]
- `tell me about the m80 aircraft , list airports and then how many canadian airlines flights use aircraft dh8`
  - reference: [(0, 5), (7, 8), (11, 18)]; rule: [(0, 18)]
- `what day of the week do flights from nashville to tacoma fly on and also flight number from houston to dallas`
  - reference: [(0, 12), (15, 20)]; rule: [(0, 20)]
- `list la , list ground transportation in baltimore and what meals are available on dl 468 which al arrives in san francisco at 950 am`
  - reference: [(0, 1), (3, 7), (9, 24)]; rule: [(0, 7), (9, 24)]
- `what class is fare code q and also list airports`
  - reference: [(0, 5), (8, 9)]; rule: [(0, 9)]
- `what meals are served on american flight 665 673 from milwaukee to seattle and also how many canadian airlines international flights use j31`
  - reference: [(0, 12), (15, 22)]; rule: [(0, 13), (15, 22)]
- `what is the capacity of the 73s , what day of the week do flights from nashville to tacoma fly on and then what are the departure times from detroit to westchester county`
  - reference: [(0, 6), (8, 20), (23, 32)]; rule: [(0, 6), (8, 32)]
- `determine the type of aircraft used on a flight from cleveland to dallas that leaves before noon , how much does it cost to fly from columbus to st. louis round trip on twa and also how much is the limousine service in boston`
  - reference: [(0, 16), (18, 33), (36, 43)]; rule: [(0, 16), (18, 34), (36, 43)]
- `list california airports , which flights travel from cleveland to indianapolis on april fifth and also what are the fares for ground transportation in denver`
  - reference: [(0, 2), (4, 13), (16, 24)]; rule: [(0, 14), (16, 24)]
- `what does fare code f mean , i need a ticket from nashville tennessee to seattle and then which airport is closest to ontario california`
  - reference: [(0, 5), (7, 15), (18, 24)]; rule: [(0, 24)]

## Rule-based correct, BERT wrong
- `open fadl shaker on spotify and play a melody starting with the newest`
  - reference: [(0, 12)]; BERT: [(0, 4), (6, 12)]; rule: [(0, 12)]
- `21 weeks from now elinor crystal turner and nita want to eat german food at a bar in distant california`
  - reference: [(0, 19)]; BERT: [(0, 6)]; rule: [(0, 19)]

## Both wrong
- `list the arizona airport and list la`
  - reference: [(0, 3), (5, 6)]; BERT: [(0, 6)]; rule: [(0, 6)]
- `at the charlotte airport how many different types of aircraft are there for us air and st. paul to kansas city friday night`
  - reference: [(0, 14), (16, 22)]; BERT: [(0, 22)]; rule: [(0, 22)]
- `what day of the week do flights from nashville to tacoma fly on and cleveland to miami on wednesday arriving before 4 pm`
  - reference: [(0, 12), (14, 22)]; BERT: [(0, 13), (14, 22)]; rule: [(0, 22)]
- `list all flights from cleveland to nashville , how much is limousine service in los angeles and and now show me ground transportation that i could get in boston late night`
  - reference: [(0, 6), (8, 15), (17, 30)]; BERT: [(0, 6), (8, 15)]; rule: [(0, 6), (8, 30)]
