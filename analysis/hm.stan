data {    
    int<lower=1> N;    
    int<lower=1> G;
    int<lower=1> V;
    int<lower=1> R;
    real x[N];    
}

parameters {
    real gamma[G];   
    real[V] beta[G];
    real[V] l[G];
    real<lower=0> tau;
    real<lower=0> sigma;
}

model {
    tau ~ cauchy(0, 3);
      
     
    for(i in 1:G) {
    	
    	gamma[i] ~ normal(0, tau);
    	
    	for (j in 1:V) {
    		
    		l[i, j] ~ gamma(1, 1);
    		beta[i, j] ~ normal(gamma[i], tau);
    		
    		for (k in 1:R) {
    			x[i] ~ poisson(l[i, j] * exp(beta[i, j]))
    		}
    	}    	
    }
}
