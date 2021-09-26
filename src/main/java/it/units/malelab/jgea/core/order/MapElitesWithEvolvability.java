package it.units.malelab.jgea.core.order;

import it.units.malelab.jgea.core.util.Pair;
import org.checkerframework.checker.units.qual.A;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;



public class MapElitesWithEvolvability<T> extends MapElites<T>{

    protected HashMap<List<Integer>, Pair<T, Integer>> evolvabilityArchive;
    protected BiFunction<T, Integer, double[]> setEvo;
    protected double minimumFactor = 0d;
    public ArrayList<T> lastAddedEvolvability;

    public MapElitesWithEvolvability(List<Integer> size, List<Double> min, List<Double> max, Boolean maximize, Function<T, List<Double>> descriptor, PartialComparator<? super T> comparator, Function<T, Double> helper, BiFunction<T, Integer, double[]> setEvo) {
        super(size, min, max, maximize, descriptor, comparator, helper);
        evolvabilityArchive = new HashMap<>();
        this.setEvo = setEvo;

    }

    public void addAll(List<T> pops, List<List<T>> children){
        lastAddedEvolvability = new ArrayList();
        lastAddedPerformance = new ArrayList();
        for (int i = 0; i < pops.size(); i++) {
            add(pops.get(i), children.get(i));
        }
        updateMinimumFactor();
    }

    public List<T> getEvolvabilityPopulation(){
        List<T> pp = new ArrayList<>();
        for(Pair<T,Integer> p:this.evolvabilityArchive.values()){
            pp.add(p.first());
        }
        return pp;
    }

    public double getCrowednessEvolvability(T individual){
        List<Integer> indexes = index(individual);
        int crowed =0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        if (i != 0 && j != 0 && k != 0 && l != 0) {
                            List<Integer> ni = indexes;
                            ni.set(0,ni.get(0)+i);
                            ni.set(1,ni.get(1)+j);
                            ni.set(2,ni.get(2)+k);
                            ni.set(3,ni.get(3)+l);
                            crowed += evolvabilityArchive.containsKey(ni)? 1:0;
                        }
                    }
                }
            }

        }
        return 1 - crowed/(Math.pow(3,4)-1);

    }

    public void add(T ind, List<T> children){

        this.add(ind);
        HashSet<List<Integer>> differentChild = new HashSet<>();
        differentChild.add(index(ind));

        for (T child: children){
            if( helper.apply(child) >= minimumFactor) {
                differentChild.add(index(child));
            }
        }
        int evolvability = differentChild.size();


        Pair<T, Integer> pair = evolvabilityArchive.get(index(ind));

        if (pair == null) {
            evolvabilityArchive.put(index(ind), Pair.of(ind,evolvability));
            setEvo.apply(ind, evolvability);
            lastAddedEvolvability.add(ind);
        }else if (pair.second()<= evolvability){

            evolvabilityArchive.put(index(ind), Pair.of(ind,evolvability));
            setEvo.apply(ind, evolvability);
            lastAddedEvolvability.add(ind);
        }
    }

    public void updateMinimumFactor(){
        this.minimumFactor = archive.values().stream().mapToDouble(ind -> helper.apply((T)ind)).average().getAsDouble();
    }


}
