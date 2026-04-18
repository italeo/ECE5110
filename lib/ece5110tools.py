class ece5110tools:
    def get_val(self, f, x):
        return f(x)
    
    def solve_bisection_rec(self, f, a, b, precision, max_steps):
        if max_steps <= 0:
            return 0, -2
        if f(a) * f(b) >= 0:
            return 0, -1
        
        if abs(f(a)) < precision:
            return a, 0
        if abs(f(b)) < precision:
            return b, 0
        
        mid = (a + b) / 2
        if f(a) + f(mid) > 0:
            return self.solve_bisection_rec(f, mid, b, precision, max_steps - 1)
        
        return self.solve_bisection_rec(f, a, mid, precision, max_steps - 1)
    

    def solve_bisection_loop(self, f, a, b, precision=1e-6, max_steps=100):
        if abs(f(a)) < precision:
            return a, 0
        if abs(f(b)) < precision:
            return b, 0
        if f(a) * f(b) > 0:
            return 0,-1
        
        for _ in range(max_steps):
            c = (a + b) / 2.0
            if abs(f(c)) < precision:
                return c, 0
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        
        return (a + b) / 2.0, -2  
    # def solve_bisection_loop(self, f, a, b, precision=1e-6, max_steps=100):
    #     if abs(f(a)) < precision:
    #         return a, 0
    #     if abs(f(b)) < precision:
    #         return b, 0
    #     if f(a) * f(b) > 0:
    #         return 0, -1
    
