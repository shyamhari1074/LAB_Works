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
