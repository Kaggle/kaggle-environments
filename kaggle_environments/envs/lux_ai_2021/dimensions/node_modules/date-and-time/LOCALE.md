# Locale

The `format()` outputs month, day of week, and meridiem (am / pm) in English, and the `parse()` assumes the passed date string is in English. Here it describes how to use other languages in these functions.

## Usage

To support `ES Modules` in the next version, the locale switching method has changed and then the old method has been deprecated.

- CommonJS:

```javascript
const date = require('date-and-time');
const fr = require('date-and-time/locale/fr');

date.locale(fr);    // French
date.format(new Date(), 'dddd D MMMM'); // => 'lundi 11 janvier'
```

- ES Modules (with transpile):

```javascript
import date from 'date-and-time';
import it from 'date-and-time/locale/it';

date.locale(it);    // Italian
date.format(new Date(), 'dddd D MMMM'); // => 'Lunedì 11 gennaio'
```

- Older browser:

When in older browser, pass the locale string as before. (no changes)

```html
<script src="/path/to/date-and-time.min.js"></script>
<script src="/path/to/locale/zh-cn.js"></script>

<script>
date.locale('zh-cn');   // Chinese
date.format(new Date(), 'MMMD日dddd');  // => '1月11日星期一'
</script>
```

### NOTE

- You have to import (or require) in advance the all locale modules that you are going to switch to.
- The locale will be actually switched after executing the `locale()`.
- You can also change the locale back to English by loading the `en` module:

```javascript
import en from 'date-and-time/locale/en';

date.locale(en);
```

### FYI

The following (old) methods are deprecated. In the next version it won't be able to use them.

- CommonJS:

```javascript
const date = require('date-and-time');
require('date-and-time/locale/fr');

date.locale('fr');  // French
date.format(new Date(), 'dddd D MMMM'); // => 'lundi 11 janvier'
```

- ES Modules (with transpile):

```javascript
import date from 'date-and-time';
import 'date-and-time/locale/it';

date.locale('it');  // Italian
date.format(new Date(), 'dddd D MMMM'); // => 'Lunedì 11 gennaio'
```

## Supported locale List

At this time, it supports the following locales:

```text
Arabic (ar)
Azerbaijani (az)
Bengali (bn)
Burmese (my)
Chinese (zh-cn)
Chinese (zh-tw)
Czech (cs)
Danish (dk)
Dutch (nl)
English (en)
French (fr)
German (de)
Greek (el)
Hindi (hi)
Hungarian (hu)
Indonesian (id)
Italian (it)
Japanese (ja)
Javanese (jv)
Korean (ko)
Persian (fa)
Polish (pl)
Portuguese (pt)
Punjabi (pa-in)
Romanian (ro)
Russian (ru)
Serbian (sr)
Spanish (es)
Thai (th)
Turkish (tr)
Ukrainian (uk)
Uzbek (uz)
Vietnamese (vi)
```
