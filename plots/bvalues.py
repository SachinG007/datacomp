# a: 1.0260 | b: -0.0574 | c: 31.4791 | d: 0.0337
# a: 1.0250 | b: -0.0637 | c: 31.9264 | d: 0.0322
# a: 1.0210 | b: -0.0538 | c: 27.7887 | d: 0.0291
# a: 1.0143 | b: -0.0394 | c: 21.2526 | d: 0.0234
# a: 1.0078 | b: -0.0254 | c: 14.4693 | d: 0.0175
# a: 1.0027 | b: -0.0152 | c: 8.7232 | d: 0.0126

#create a list of a,b,c,d from above
a = [1.0250, 1.0260, 1.0210, 1.0143, 1.0078, 1.0027]
b = [ -0.0637, -0.057, -0.0538, -0.0394, -0.0254, -0.0152]
# c = [31.9264, 31.4791, 27.7887, 21.2526, 14.4693, 8.7232]
#c is a list of all 3 
c = [2, 2, 2, 2, 2, 2]
d = [0.0322, 0.0337, 0.0291, 0.0234, 0.0175, 0.0126]

def power(x, power):
    if x==0:
        return 0
    return x**power

# for epoch e, calculate sum_{i=0}^{M} sum_{k=1}^{e}a_i*[k**b_i - (k-1)**b_i]*base**(k/c)
compute = 32_000_000
samples_per_bucket = 12_800_000
base = 0.5
avg_utility_dict = {}
for m in range(1,7):
    avg_utility_m = 0
    epochs = compute // samples_per_bucket
    remaining = compute / samples_per_bucket
    print("remaining", remaining)
    if remaining < 0.0001:
        remaining = 0
    
    bucket=0
    for k in range(1,epochs+1):
        bucket = bucket % m
        repeat = (k-1) // m
        print("m", m," Bucket", bucket, " repeat", repeat)
        avg_utility_m += a[bucket] * (power(k,b[bucket]) - power(k-1,b[bucket]))*base**((repeat)/c[bucket])
        bucket = bucket + 1

    print("avg utility m", avg_utility_m)
    if remaining > 0.01:
        bucket = bucket % m
        avg_utility_m += a[bucket] * (power(remaining,b[bucket]) - power(epochs,b[bucket]))*base**(epochs/c[bucket])

    avg_utility_dict[m] = avg_utility_m

print("avg utility dict", avg_utility_dict)