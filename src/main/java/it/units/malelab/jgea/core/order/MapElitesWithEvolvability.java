package it.units.malelab.jgea.core.order;

import it.units.malelab.jgea.core.util.Pair;

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
