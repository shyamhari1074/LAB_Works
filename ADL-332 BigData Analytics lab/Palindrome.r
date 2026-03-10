is_palindrome <- function(x) {
  reversed <- paste(rev(strsplit(x, "")[[1]]), collapse = "")
  if (x == reversed) {
    return("Palindrome")
  } else {
    return("Not Palindrome")
  }
}
# Test
is_palindrome("madam")
is_palindrome("racecar")
is_palindrome("hello")

/// simple code 

is_palindrome <- function(word) {
  reversed <- paste(rev(strsplit(word, "")[[1]]), collapse="")
  return(word == reversed)
}
is_palindrome("madam")


//// 
word <- readline("Enter a word: ")
flag <- TRUE
n <- nchar(word)
for (i in 1:(n/2)) {
  if (substr(word, i, i) != substr(word, n - i + 1, n - i + 1)) {
    flag <- FALSE
    break
  }
}
if (flag) {
  print("Word Palindrome")
} else {
  print("Not a palindrome")
}
