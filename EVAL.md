# Acceptable Model Responses

Direct answers should be enclosed by a LaTeX `\boxed{}` tag, but this is a soft requirement. Any formats that include a reasonable numeric answer (within 1e-2) should be accepted.
```
>>> extract_final_answer(r"The answer is \boxed{42}")
'42'
>>> extract_final_answer("The final answer is: 3.14")
'3.14'
>>> extract_final_answer("Just a number 100 in text")
'100'
>>> extract_final_answer(None)
'NOT_FOUND'
```

As mentioned, rounding within 1e-2 is accepted and case-sensitivity is not strictly enforced.
```
>>> verify("10011000", "10011000")
True
>>> verify("10011000", "10011001")
False
>>> verify("24.64", "24.6401")
True
>>> verify("XLVII", "xlvii")
True
>>> verify("11011", "00011011")
False
```