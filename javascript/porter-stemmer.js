// porter-stemmer.js

function isVowel(char) {
    return /[aeiou]/i.test(char);
  }
  
  function getStartConsonantSequence(word) {
    const match = word.match(/^[^aeiou]+/);
    return match ? match[0].length : 0;
  }
  
  function getStartVowelSequence(word) {
    const match = word.match(/^[aeiou]+/);
    return match ? match[0].length : 0;
  }
  
  function containsVowel(word) {
    return isVowel(word.charAt(0)) || word.substring(1).search(/[aeiou]/) !== -1;
  }
  
  function isShortWord(word) {
    return getStartConsonantSequence(word) === 1 && getStartVowelSequence(word.substring(1)) === 1;
  }
  
  function isShortSyllable(word) {
    return (
      word.length === 2 &&
      isVowel(word.charAt(0)) &&
      !isVowel(word.charAt(1))
    );
  }
  
  function removeEnding(word, ending) {
    if (word.endsWith(ending)) {
      return word.substring(0, word.length - ending.length);
    }
    return word;
  }
  
  function replace(word, ending, replacement) {
    return removeEnding(word, ending) + replacement;
  }
  
  function isStep1aException(word) {
    return (
      isShortWord(word.substring(0, word.length - 1)) ||
      isShortSyllable(word.substring(0, word.length - 2))
    );
  }
  
  function processStep1a(word) {
    if (word.endsWith("sses")) {
      return word.substring(0, word.length - 2);
    }
    if (word.endsWith("ies") && word.length > 4) {
      return word.substring(0, word.length - 2);
    }
    if (word.endsWith("ss")) {
      return word;
    }
    if (word.endsWith("s") && !isStep1aException(word)) {
      return word.substring(0, word.length - 1);
    }
    return word;
  }
  
  function processStep1b(word) {
    if (word.endsWith("eed") && getStartConsonantSequence(word.substring(0, word.length - 3)) > 0) {
      return word.substring(0, word.length - 1);
    }
    if (word.endsWith("ed") && containsVowel(word.substring(0, word.length - 2))) {
      word = removeEnding(word, "ed");
      if (/(at|bl|iz)$/.test(word)) {
        return word + "e";
      }
      if (/([^aeiouylsz])\1$/.test(word)) {
        return word.substring(0, word.length - 1);
      }
      if (isShortWord(word)) {
        return word + "e";
      }
      return word;
    }
    if (word.endsWith("ing") && containsVowel(word.substring(0, word.length - 3))) {
      word = removeEnding(word, "ing");
      if (/(at|bl|iz)$/.test(word)) {
        return word + "e";
      }
      if (/([^aeiouylsz])\1$/.test(word)) {
        return word.substring(0, word.length - 1);
      }
      if (isShortWord(word)) {
        return word + "e";
      }
      return word;
    }
    return word;
  }
  
  function processStep1c(word) {
    if (word.endsWith("y") && containsVowel(word.substring(0, word.length - 1))) {
      return word.substring(0, word.length - 1) + "i";
    }
    return word;
  }
  
  function processStep2(word) {
    const step2Rules = [    ['ational', 'ate'],
      ['tional', 'tion'],
      ['enci', 'ence'],
      ['anci', 'ance'],
      ['izer', 'ize'],
      ['bli', 'ble'],
      ['alli', 'al'],
      ['entli', 'ent'],
      ['eli', 'e'],
      ['ousli', 'ous'],
      ['ization', 'ize'],
      ['ation', 'ate'],
      ['ator', 'ate'],
      ['alism', 'al'],
      ['iveness', 'ive'],
      ['fulness', 'ful'],
      ['ousness', 'ous'],
      ['aliti', 'al'],
      ['iviti', 'ive'],
      ['biliti', 'ble'],
      ['logi', 'log']
    ];
  
    for (const [ending, replacement] of step2Rules) {
      if (word.endsWith(ending)) {
        const stem = removeEnding(word, ending);
        if (getStartConsonantSequence(stem) > 0) {
          return stem + replacement;
        }
      }
    }
  
    return word;
  }
  
  function processStep3(word) {
    const step3Rules = [
      ['icate', 'ic'],
      ['ative', ''],
      ['alize', 'al'],
      ['iciti', 'ic'],
      ['ical', 'ic'],
      ['ful', ''],
      ['ness', '']
    ];
  
    for (const [ending, replacement] of step3Rules) {
      if (word.endsWith(ending)) {
        const stem = removeEnding(word, ending);
        if (getStartConsonantSequence(stem) > 0) {
          return stem + replacement;
        }
      }
    }
  
    return word;
  }
  
  function processStep4(word) {
    const step4Rules = [
      'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement',
      'ment', 'ent', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize'
    ];
  
    for (const ending of step4Rules) {
      if (word.endsWith(ending)) {
        const stem = removeEnding(word, ending);
        if (getStartConsonantSequence(stem) > 1) {
          return stem;
        }
      }
    }
  
    return word;
  }
  
  function processStep5a(word) {
    if (word.endsWith("e")) {
      const stem = word.substring(0, word.length - 1);
      if (getStartConsonantSequence(stem) > 1) {
        return stem;
      }
      if (getStartConsonantSequence(stem) === 1 && !isShortSyllable(stem)) {
        return stem;
      }
    }
    return word;
  }
  
  function processStep5b(word) {
    if (getStartConsonantSequence(word) > 1 && isShortWord(word) && word.endsWith("ll")) {
      return word.substring(0, word.length - 1);
    }
    return word;
  }
  
  function stemWord(word) {
    word = word.toLowerCase();
    word = processStep1a(word);
    word = processStep1b(word);
    word = processStep1c(word);
    word = processStep2(word);
    word = processStep3(word);
    word = processStep4(word);
    word = processStep5a(word);
    word = processStep5b(word);
    return word;
  }